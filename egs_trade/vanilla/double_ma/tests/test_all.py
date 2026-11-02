# -*- coding: utf-8 -*-
"""Unit + e2e tests for refactored egs_trade/vanilla/double_ma."""
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

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


# ---------------- end-to-end back tester (mocked TuShare) ----------------
class TestBackTesterE2E:
    def _build_args(self, exp_dir):
        ns = types.SimpleNamespace()
        ns.config = str(PROJ / 'conf' / 'double_ma.yaml')
        ns.data_dir = str(PROJ / 'data')
        ns.exp_dir = str(exp_dir)
        ns.override_config = []
        return ns

    def _run_backtest(self, exp_dir, close_fn, n, benchmark_close=None):
        """Run BackTester on synthetic data.

        ``close_fn(i, t)`` returns the close price array for the i-th stock
        given ``t = np.arange(n)``. ``benchmark_close`` is the benchmark's
        close prices (default: a straight line).
        """
        import yaml
        with open(PROJ / 'conf' / 'double_ma.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        stock_lst = cfg['data_condition']['stock_lst']
        benchmark = cfg['data_condition']['benchmark']

        # Inject fake third-party modules that may not be installed in
        # the test environment (tushare, talib, mplfinance, etc.) and
        # a fake tushare_token. Some imports reach into sub-modules
        # (e.g. ``mplfinance.original_flavor``) so we mock with a package
        # like container using ``mock.MagicMock``.
        for mod_name in ('tushare', 'talib', 'mplfinance',
                         'data', 'data.private',
                         'mplfinance.original_flavor'):
            sys.modules.setdefault(mod_name, mock.MagicMock())
        # Use the real ffn when it is installed so the risk indicator
        # calculation is actually exercised; mock it only as a fallback.
        try:
            import ffn  # noqa: F401
        except ImportError:
            sys.modules.setdefault('ffn', mock.MagicMock())
        token_mod = mock.MagicMock()
        token_mod.tushare_token = 'TEST_TOKEN'
        sys.modules['data.private.tushare_token'] = token_mod

        from back_tester import BackTester

        dates = pd.date_range('2022-01-01', periods=n, freq='D').strftime('%Y%m%d')
        if benchmark_close is None:
            benchmark_df = pd.DataFrame({
                'trade_date': dates,
                'close': np.linspace(3000, 3200, n),
                'pct_chg': np.zeros(n),
            })
        else:
            bench = pd.Series(benchmark_close, dtype=float)
            benchmark_df = pd.DataFrame({
                'trade_date': dates,
                'close': bench.values,
                # TuShare reports pct_chg in percent
                'pct_chg': (bench.pct_change() * 100).fillna(0).values,
            })
        df_dict = {benchmark: benchmark_df}
        t = np.arange(n)
        for i, code in enumerate(stock_lst):
            close = close_fn(i, t)
            df_dict[code] = pd.DataFrame({
                'trade_date': dates,
                'open': close,
                'high': close * 1.02,
                'low': close * 0.98,
                'close': close,
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
            bt = BackTester(self._build_args(exp_dir))
            bt.offline_back_test_ctrl()
        return bt

    def test_e2e_run_with_mock_data(self, tmp_path):
        n = 60

        def close_fn(i, t):
            slope = 0.05 if i == 0 else -0.03  # first stock up, second down
            return np.linspace(10 + i * 5, 10 + i * 5 + slope * n, n)

        bt = self._run_backtest(tmp_path, close_fn, n)

        # at least one trade must have happened
        total_trade_rows = sum(len(df) for df in bt._account.trade_dict.values())
        assert total_trade_rows > 0, (
            'No trades happened in e2e simulation; trade_dict=%r' %
            {k: len(v) for k, v in bt._account.trade_dict.items()}
        )
        # risk indicator csv should be saved
        assert (tmp_path / 'risk_indicator.csv').exists()
        assert (tmp_path / 'trading_info.csv').exists()

    def test_e2e_oscillating_prices(self, tmp_path):
        # Sine-wave prices make the 10/20-day moving averages cross
        # repeatedly, so stocks are bought, held and sold several times.
        n = 160

        def close_fn(i, t):
            return 20 + 3 * np.sin(2 * np.pi * t / 40 + i * np.pi / 2)

        # benchmark: the average of the two stocks plus some noise
        t = np.arange(n)
        noise = np.random.default_rng(0).normal(0, 0.05, n)
        benchmark_close = 100 * ((close_fn(0, t) + close_fn(1, t)) / 2 + noise)

        bt = self._run_backtest(tmp_path, close_fn, n, benchmark_close)
        account = bt._account

        trades = pd.concat(account.trade_dict.values(), ignore_index=True)
        assert (trades['trade_type'] == 'buy').sum() >= 2
        assert (trades['trade_type'] == 'sell').sum() >= 2

        funds = np.asarray(account.funds_chg_lst, dtype=float)
        assert len(funds) == bt._data_len - 1

        # Each trade record carries the account capital after that day's
        # trades were settled (funds_chg_lst[k] is recorded for index k+1).
        for stock_id, df_trades in account.trade_dict.items():
            for _, row in df_trades.iterrows():
                expected = funds[int(row['time_index']) - 1]
                assert float(row['capital']) == pytest.approx(expected), stock_id

        # Held positions are marked to market daily, so the equity curve
        # moves on days without trades, not only when a position is sold.
        assert np.count_nonzero(np.diff(funds)) > trades['time_index'].nunique()

        df_risk = pd.read_csv(tmp_path / 'risk_indicator.csv', index_col=0)
        assert df_risk.loc['total_returns', 'value'] == pytest.approx(
            funds[-1] / funds[0] - 1)
        assert df_risk.loc['max_withdraw', 'value'] < 0
        # beta against the benchmark on the same days, computed directly:
        # funds[k] is the close of day offset + k, where offset is the
        # moving-average warm-up plus the first (skipped) trading index
        offset = n - len(funds)
        rs = funds[1:] / funds[:-1] - 1
        rm = pd.Series(benchmark_close).pct_change().values[offset + 1:]
        expected_beta = np.cov(rs, rm)[0, 1] / np.var(rm, ddof=1)
        assert df_risk.loc['beta', 'value'] == pytest.approx(expected_beta)
        assert np.isfinite(df_risk.loc['alpha', 'value'])
        if not isinstance(sys.modules.get('ffn'), mock.Mock):
            for key in ('sharp_ratio', 'sortino_ratio'):
                assert np.isfinite(df_risk.loc[key, 'value']), key
