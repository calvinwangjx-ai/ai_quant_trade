# -*- coding: utf-8 -*-
"""Unit + e2e tests for refactored egs_trade/vanilla/double_ma."""
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.back_test.account_info import Account
from src.back_test.cal_fee import calculate_fee
from src.back_test.trading_ctrl import order_value
from src.portfolio.capital_allocation import equal_allocation
from src.rules.timing_ctrl.moving_average import double_ma_timing


# ---------------- pure-function unit tests ----------------
class TestDoubleMaTiming:
    def test_buy_when_short_above_long_and_no_hold(self):
        assert double_ma_timing(10.0, 9.0, hold=False) == 'buy'

    def test_sell_when_short_below_long_and_hold(self):
        assert double_ma_timing(9.0, 10.0, hold=True) == 'sell'

    def test_no_action_when_short_above_long_but_holding(self):
        assert double_ma_timing(10.0, 9.0, hold=True) == ''

    def test_no_action_when_short_below_long_and_no_hold(self):
        assert double_ma_timing(9.0, 10.0, hold=False) == ''


class TestAccount:
    def test_initial_capital(self):
        a = Account(100_000)
        assert a.cash == 100_000
        assert a.total_capital == 100_000
        assert a.get_total_capital() == 100_000

    def test_total_capital_with_position(self):
        a = Account(50_000)
        a.pos_dict['600000.SH'] = {'pos_num': 100, 'price': 11.0}
        # cash 50_000 + position 100 * 11 = 1_100 = 51_100
        assert a.get_total_capital() == 51_100

    def test_open_position_marked_to_market(self):
        a = Account(50_000)
        a.pos_dict['600000.SH'] = {'pos_num': 100, 'price': 11.0}
        # value moves with the current price
        assert a.get_total_capital({'600000.SH': 12.5}) == 50_000 + 100 * 12.5
        assert a.total_capital == 51_250
        assert a.get_total_capital({'600000.SH': 9.0}) == 50_000 + 100 * 9.0
        # cost basis is left untouched
        assert a.pos_dict['600000.SH']['price'] == 11.0
        # no prices -> entry-price valuation (existing behaviour)
        assert a.get_total_capital() == 51_100

    def test_mark_to_market_falls_back_to_entry_price(self):
        a = Account(0)
        a.pos_dict['A'] = {'pos_num': 10, 'price': 5.0}
        a.pos_dict['B'] = {'pos_num': 20, 'price': 2.0}
        # only A has a quote; B keeps its entry price; unheld C is ignored
        assert a.get_total_capital({'A': 6.0, 'C': 100.0}) == 10 * 6.0 + 20 * 2.0


class TestAddLog:
    def test_exception_exits_non_zero(self):
        import pytest
        from src.tools.log.log_util import addlog

        @addlog(name='boom')
        def boom():
            raise ValueError('boom')

        with pytest.raises(SystemExit) as exc_info:
            boom()
        assert exc_info.value.code != 0


class TestRiskIndicatorRealFfn:
    def test_ratios_with_integer_indexed_equity(self, tmp_path):
        """Real ffn cannot infer frequency from a plain list; must not crash."""
        import pytest
        ffn = pytest.importorskip('ffn')
        if isinstance(ffn, mock.MagicMock):
            pytest.skip('ffn is mocked in this session')
        from src.back_test.risk_indicator import cal_risk_indicator

        rng = np.random.default_rng(0)
        equity = list(100_000 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)))
        df_trade = pd.DataFrame({'time_index': [5, 20, 40],
                                 'capital': [99_000.0, 101_000.0, 103_000.0],
                                 'trade_detail': ['b', 's', 'b']})
        df_res = cal_risk_indicator(100_000, 0.04, equity, df_trade,
                                    pd.DataFrame(), str(tmp_path))

        for key in ('sharp_ratio', 'sortino_ratio', 'max_withdraw'):
            assert np.isfinite(float(df_res.loc[key, 'value'])), key
        assert (tmp_path / 'risk_indicator.csv').exists()


class TestCalculateFee:
    def test_buy_commission_only(self):
        price_dict = pd.Series({'close': 10.0, 'low': 9.5, 'high': 10.5})
        order_cost = {
            'open_commission': 0.0003,
            'close_commission': 0.0003,
            'min_commission': 5.0,
            'close_tax': 0.001,
            'slippage_fee': 0.0,
        }
        fee = calculate_fee(price_dict, 100, order_cost, 'buy')
        # 100 * 0.0003 = 0.03 < 5, so commission = 5
        assert fee == 5.0
        assert fee >= order_cost['min_commission']

    def test_sell_with_tax(self):
        price_dict = pd.Series({'close': 10.0, 'low': 9.5, 'high': 10.5})
        order_cost = {
            'open_commission': 0.0003,
            'close_commission': 0.0003,
            'min_commission': 5.0,
            'close_tax': 0.001,
            'slippage_fee': 0.0,
        }
        fee = calculate_fee(price_dict, 100, order_cost, 'sell')
        # commission = 5 (min), tax = 0.001 * 10 * 100 = 1
        assert fee == 5.0 + 1.0


class TestOrderValue:
    def test_buy_updates_cash_and_position(self):
        a = Account(20_000)
        price_dict = pd.Series({'close': 10.0, 'low': 9.5, 'high': 10.5})
        order_cost = {
            'open_commission': 0.0003,
            'close_commission': 0.0003,
            'min_commission': 5.0,
            'close_tax': 0.001,
            'slippage_fee': 0.0,
            'trade_lim': 100,
        }
        order_type, pos, _ = order_value(a, '600000.SH', price_dict, 'buy',
                                          order_funds=10_000, order_cost=order_cost)
        # pos = int(10_000 / 10 / 100) * 100 = 1000, fee (commission>=5) is added,
        # so rest is negative and the order is rejected by the safety check.
        # The function returns the candidate pos but signals rejection by
        # leaving order_type empty and not touching account state.
        assert order_type == ''
        assert pos == 1000  # candidate quantity before cash check
        assert '600000.SH' not in a.pos_dict
        # cash must remain untouched
        assert a.cash == 20_000

    def test_sell_clears_position(self):
        a = Account(20_000)
        a.pos_dict['600000.SH'] = {'pos_num': 100, 'price': 9.0}
        price_dict = pd.Series({'close': 10.0, 'low': 9.5, 'high': 10.5})
        order_cost = {
            'open_commission': 0.0003,
            'close_commission': 0.0003,
            'min_commission': 5.0,
            'close_tax': 0.001,
            'slippage_fee': 0.0,
            'trade_lim': 100,
        }
        order_type, pos, _ = order_value(a, '600000.SH', price_dict, 'sell',
                                          order_funds=0, order_cost=order_cost)
        assert order_type == 'sell'
        assert pos == 100
        assert '600000.SH' not in a.pos_dict
        assert a.cash > 20_000  # made profit


class TestEqualAllocation:
    def test_allocates_evenly(self):
        a = Account(90_000)
        # no positions held, want to pick 3 stocks => 30_000 each
        assert equal_allocation(a, select_num=3) == 30_000.0

    def test_zero_when_full(self):
        a = Account(10_000)
        a.pos_dict = {f'stk{i}': {'pos_num': 1, 'price': 1.0} for i in range(3)}
        assert equal_allocation(a, select_num=3) == 0


# ---------------- risk indicator: beta / alpha (real ffn) ----------------
class TestRiskIndicatorBeta:
    """Needs the real ``ffn``; runs before the mocked e2e test, which
    injects a fake ``ffn`` into ``sys.modules``."""

    @staticmethod
    def _cal_risk_indicator():
        import pytest
        pytest.importorskip('ffn')
        if isinstance(sys.modules.get('ffn'), mock.MagicMock):
            pytest.skip('ffn is mocked in this session')
        from src.back_test.risk_indicator import cal_risk_indicator
        return cal_risk_indicator

    @staticmethod
    def _benchmark(n=100, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range('2022-01-03', periods=n).strftime('%Y%m%d')
        return pd.DataFrame({
            'trade_date': dates,
            'close': 3000 * np.cumprod(1 + rng.normal(0, 0.01, n)),
            'pct_chg': rng.normal(0.05, 1.0, n),  # percent, like tushare
        })

    @staticmethod
    def _trades(capital_list):
        return pd.DataFrame({'time_index': [0, 1],
                             'capital': capital_list[:2],
                             'trade_detail': ['buy', 'sell']})

    def test_double_benchmark_returns_gives_beta_two(self, tmp_path):
        cal_risk_indicator = self._cal_risk_indicator()
        df_index = self._benchmark()
        df_index_before = df_index.copy(deep=True)

        # strategy only covers bars after a 20-bar warm-up (like the MA
        # drop + skipped bar 0), so alignment must be by trade date
        start = 20
        bench_ret = df_index['pct_chg'].to_numpy()[start:] / 100
        strat_ret = 2 * bench_ret
        strat_ret[0] = 0.0  # first bar has no previous capital
        capital_list = list(100_000 * np.cumprod(1 + strat_ret))
        trade_dates = df_index['trade_date'].iloc[start:].tolist()

        base_rise = 0.04
        df = cal_risk_indicator(100_000, base_rise, capital_list,
                                self._trades(capital_list), df_index,
                                str(tmp_path), trade_dates=trade_dates)

        beta = float(df.loc['beta', 'value'])
        assert abs(beta - 2.0) < 1e-9

        # alpha uses compounded fractional returns over the aligned bars
        rise = np.prod(1 + strat_ret[1:]) - 1
        sign_rise = np.prod(1 + bench_ret[1:]) - 1
        expected_alpha = rise - (base_rise + 2.0 * (sign_rise - base_rise))
        assert abs(float(df.loc['alpha', 'value']) - expected_alpha) < 1e-9

        # caller's benchmark must not be modified
        pd.testing.assert_frame_equal(df_index, df_index_before)

    def test_zero_variance_benchmark_gives_nan_beta(self, tmp_path):
        cal_risk_indicator = self._cal_risk_indicator()
        df_index = self._benchmark()
        df_index['pct_chg'] = 0.0
        capital_list = list(np.linspace(100_000, 110_000, len(df_index)))
        df = cal_risk_indicator(100_000, 0.04, capital_list,
                                self._trades(capital_list), df_index,
                                str(tmp_path),
                                trade_dates=df_index['trade_date'].tolist())
        assert np.isnan(float(df.loc['beta', 'value']))

    def test_benchmark_without_trade_dates_raises(self):
        # call the undecorated helper: @addlog turns exceptions into sys.exit
        import pytest
        self._cal_risk_indicator()
        from src.back_test.risk_indicator import _cal_beta_and_rises
        df_index = self._benchmark()
        capital_list = list(np.linspace(100_000, 110_000, len(df_index)))
        with pytest.raises(ValueError, match='trade_dates'):
            _cal_beta_and_rises(capital_list, None, df_index)
        with pytest.raises(ValueError, match='same length'):
            _cal_beta_and_rises(capital_list,
                                df_index['trade_date'].tolist()[1:], df_index)


# ---------------- end-to-end back tester (mocked TuShare) ----------------
class TestBackTesterE2E:
    def _build_args(self, exp_dir):
        ns = types.SimpleNamespace()
        ns.config = str(PROJ / 'conf' / 'double_ma.yaml')
        ns.data_dir = str(PROJ / 'data')
        ns.exp_dir = str(exp_dir)
        ns.override_config = []
        return ns

    def test_e2e_run_with_mock_data(self, tmp_path):
        import yaml
        with open(PROJ / 'conf' / 'double_ma.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        stock_lst = cfg['data_condition']['stock_lst']
        benchmark = cfg['data_condition']['benchmark']

        # Inject fake third-party modules that may not be installed in
        # the test environment (tushare, ffn, talib, mplfinance, etc.) and
        # a fake tushare_token. Some imports reach into sub-modules
        # (e.g. ``mplfinance.original_flavor``) so we mock with a package
        # like container using ``mock.MagicMock``.
        for mod_name in ('tushare', 'ffn', 'talib', 'mplfinance',
                         'data', 'data.private',
                         'mplfinance.original_flavor'):
            sys.modules.setdefault(mod_name, mock.MagicMock())
        token_mod = mock.MagicMock()
        token_mod.tushare_token = 'TEST_TOKEN'
        sys.modules['data.private.tushare_token'] = token_mod

        from back_tester import BackTester

        n = 60
        dates = pd.date_range('2022-01-01', periods=n, freq='D').strftime('%Y%m%d')
        benchmark_df = pd.DataFrame({
            'trade_date': dates,
            'close': np.linspace(3000, 3200, n),
            'pct_chg': np.zeros(n),
        })
        df_dict = {benchmark: benchmark_df}
        for i, code in enumerate(stock_lst):
            slope = 0.05 if i == 0 else -0.03  # first stock up, second down
            base = np.linspace(10 + i * 5, 10 + i * 5 + slope * n, n)
            df_dict[code] = pd.DataFrame({
                'trade_date': dates,
                'open': base,
                'high': base * 1.02,
                'low': base * 0.98,
                'close': base,
                'vol': np.ones(n) * 1_000_000,
            })

        # Replace TuShareData with a fake class that only implements the
        # methods BackTester actually calls.
        class FakeTuShareData:
            def __init__(self):
                pass
            def get_df_data(self, *args, **kwargs):
                return df_dict

        # Patch at the call site (back_tester) because it imported the
        # symbol into its own namespace.
        with mock.patch('back_tester.TuShareData', FakeTuShareData), \
             mock.patch('back_tester.plot_trades_on_capital'), \
             mock.patch('back_tester.plot_trades_on_k_line'), \
             mock.patch('back_tester.show_plt'):
            args = self._build_args(tmp_path)
            bt = BackTester(args)
            bt.offline_back_test_ctrl()

        # at least one trade must have happened
        total_trade_rows = sum(len(df) for df in bt._account.trade_dict.values())
        assert total_trade_rows > 0, (
            'No trades happened in e2e simulation; trade_dict=%r' %
            {k: len(v) for k, v in bt._account.trade_dict.items()}
        )
        # daily equity is marked to market: it must change on bars without
        # trades while a position is held. With realized-only valuation the
        # number of distinct equity values is bounded by (trade days + 1).
        funds = bt._account.funds_chg_lst
        assert len(funds) == bt._data_len - 1
        n_trade_days = len(bt._account.pd_gather_trades)
        assert len(set(np.round(funds, 6))) > n_trade_days + 1, funds
        # each trade row carries the post-trade equity of the bar it traded
        # on (bars start at index 1, funds_chg_lst at 0)
        for stock_id, df_trade in bt._account.trade_dict.items():
            for t_idx, cap in zip(df_trade['time_index'], df_trade['capital']):
                assert np.isclose(cap, funds[int(t_idx) - 1]), (stock_id, t_idx)
        # risk indicator csv should be saved
        assert (tmp_path / 'risk_indicator.csv').exists()
        assert (tmp_path / 'trading_info.csv').exists()
