# -*- coding: utf-8 -*-
"""Unit + e2e tests for refactored egs_trade/vanilla/double_ma."""
import sys
import types
import warnings
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
def _load_conf():
    import yaml
    with open(PROJ / 'conf' / 'double_ma.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _make_df_dict(close_by_stock, benchmark, n):
    """Build fake TuShare data: ``close_by_stock`` maps stock id -> close array."""
    dates = pd.date_range('2022-01-01', periods=n, freq='D').strftime('%Y%m%d')
    df_dict = {benchmark: pd.DataFrame({
        'trade_date': dates,
        'close': np.linspace(3000, 3200, n),
        'pct_chg': np.zeros(n),
    })}
    for code, close in close_by_stock.items():
        df_dict[code] = pd.DataFrame({
            'trade_date': dates,
            'open': close,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'vol': np.ones(n) * 1_000_000,
        })
    return df_dict


def _run_back_tester(tmp_path, df_dict):
    """Run BackTester end to end on ``df_dict`` with TuShare/plots mocked.

    Returns ``(back_tester, caught_warnings)``.
    """
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

    # Replace TuShareData with a fake class that only implements the
    # methods BackTester actually calls.
    class FakeTuShareData:
        def __init__(self):
            pass

        def get_df_data(self, *args, **kwargs):
            return df_dict

    args = types.SimpleNamespace(
        config=str(PROJ / 'conf' / 'double_ma.yaml'),
        data_dir=str(PROJ / 'data'),
        exp_dir=str(tmp_path),
        override_config=[],
    )

    # Patch at the call site (back_tester) because it imported the
    # symbol into its own namespace.
    with mock.patch('back_tester.TuShareData', FakeTuShareData), \
         mock.patch('back_tester.plot_trades_on_capital'), \
         mock.patch('back_tester.plot_trades_on_k_line'), \
         mock.patch('back_tester.show_plt'), \
         warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bt = BackTester(args)
        bt.offline_back_test_ctrl()
    return bt, caught


def _two_stock_df_dict():
    """Two stocks whose MA crossovers happen on different bars, with
    positions held across many bars without trades."""
    cfg = _load_conf()
    stock_a, stock_b = cfg['data_condition']['stock_lst'][:2]
    benchmark = cfg['data_condition']['benchmark']
    n = 90
    t = np.arange(n)
    close_a = 10 + 0.3 * np.where(t < 45, t, 90 - t)  # up, then down
    close_b = 20 - 0.2 * np.where(t < 60, t, 120 - t)  # down, then up
    df_dict = _make_df_dict({stock_a: close_a, stock_b: close_b}, benchmark, n)
    return stock_a, stock_b, df_dict


class TestBackTesterE2E:
    def test_e2e_run_with_mock_data(self, tmp_path):
        cfg = _load_conf()
        stock_lst = cfg['data_condition']['stock_lst']
        benchmark = cfg['data_condition']['benchmark']

        n = 60
        close_by_stock = {}
        for i, code in enumerate(stock_lst):
            slope = 0.05 if i == 0 else -0.03  # first stock up, second down
            close_by_stock[code] = np.linspace(10 + i * 5, 10 + i * 5 + slope * n, n)

        bt, _ = _run_back_tester(tmp_path, _make_df_dict(close_by_stock, benchmark, n))

        # at least one trade must have happened
        total_trade_rows = sum(len(df) for df in bt._account.trade_dict.values())
        assert total_trade_rows > 0, (
            'No trades happened in e2e simulation; trade_dict=%r' %
            {k: len(v) for k, v in bt._account.trade_dict.items()}
        )
        # risk indicator csv should be saved
        assert (tmp_path / 'risk_indicator.csv').exists()
        assert (tmp_path / 'trading_info.csv').exists()

    def test_trade_capital_matches_bar_equity(self, tmp_path):
        """Each trade row's ``capital`` is the post-trade equity of its bar.

        Regression test: ``trading_ctrl`` used a chained assignment
        (``df.iloc[-1]['capital'] = ...``) that is a no-op under pandas
        Copy-on-Write, and it also overwrote the last row of stocks that did
        not trade on the bar.
        """
        stock_a, stock_b, df_dict = _two_stock_df_dict()
        bt, caught = _run_back_tester(tmp_path, df_dict)

        chained = [w for w in caught
                   if w.category.__name__ == 'ChainedAssignmentError'
                   and w.filename.endswith('back_tester.py')]
        assert not chained, [str(w.message) for w in chained]

        acct = bt._account
        # equity recorded for each bar that had a trade
        bar_equity = dict(zip(acct.pd_gather_trades['time_index'].astype(int),
                              acct.pd_gather_trades['capital'].astype(float)))
        # both stocks traded, on different bars
        assert set(acct.trade_dict) == {stock_a, stock_b}
        bars_a = set(acct.trade_dict[stock_a]['time_index'].astype(int))
        bars_b = set(acct.trade_dict[stock_b]['time_index'].astype(int))
        assert bars_a and bars_b and bars_a != bars_b

        for stock_id, df_trade in acct.trade_dict.items():
            # the last trade row is the one the reported bug left stale
            last = df_trade.iloc[-1]
            bar = int(last['time_index'])
            # funds_chg_lst starts at bar index 1
            assert float(last['capital']) == bar_equity[bar] == acct.funds_chg_lst[bar - 1]
            # every earlier trade row also holds its own bar's equity
            for _, row in df_trade.iterrows():
                assert float(row['capital']) == bar_equity[int(row['time_index'])], (
                    stock_id, int(row['time_index']))

    def test_equity_marks_open_positions_to_market(self, tmp_path):
        """On bars without trades, equity moves with the close of held stocks.

        Regression test: held positions kept their purchase price, so the
        equity curve was flat between trades.
        """
        _, _, df_dict = _two_stock_df_dict()
        bt, _ = _run_back_tester(tmp_path, df_dict)
        acct = bt._account
        # funds_chg_lst[k] is the equity at the close of bar k + 1
        equity = {bar: acct.funds_chg_lst[bar - 1] for bar in range(1, bt._data_len)}

        trades_by_bar = {}
        for stock_id, df_trade in acct.trade_dict.items():
            for _, row in df_trade.iterrows():
                trades_by_bar.setdefault(int(row['time_index']), []).append(
                    (stock_id, row['trade_type'], int(row['pos'])))

        holdings = {}  # stock id -> shares held after the previous bar
        checked, moved = 0, 0
        for bar in range(1, bt._data_len):
            if bar in trades_by_bar:
                for stock_id, trade_type, pos in trades_by_bar[bar]:
                    if trade_type == 'buy':
                        holdings[stock_id] = pos
                    else:
                        holdings.pop(stock_id)
                continue
            if bar == 1 or not holdings:
                continue
            pnl = sum(pos * (bt._df_dict[sid]['close'].iloc[bar]
                             - bt._df_dict[sid]['close'].iloc[bar - 1])
                      for sid, pos in holdings.items())
            assert equity[bar] == pytest.approx(equity[bar - 1] + pnl), bar
            checked += 1
            moved += pnl != 0

        # the scenario must actually exercise held, non-trading bars
        assert checked > 10 and moved > 10, (checked, moved)
        assert acct.get_total_capital() == pytest.approx(equity[bt._data_len - 1])
