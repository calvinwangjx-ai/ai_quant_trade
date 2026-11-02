# -*- coding: utf-8 -*-
"""Unit tests for beta / alpha inputs in src/back_test/risk_indicator.py."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# ffn is only needed for the non-beta indicators; stub it if it is missing
try:
    import ffn  # noqa: F401
except ImportError:
    sys.modules.setdefault('ffn', mock.MagicMock())

from src.back_test.risk_indicator import (align_returns, annualize_return,
                                          cal_beta, cal_risk_indicator)


def _synthetic_pair(n=250, k=1.5, capital=100_000.0, seed=0):
    """Benchmark with random daily returns and a strategy whose daily
    return is exactly ``k`` times the benchmark's, so its beta is ``k``."""
    rng = np.random.default_rng(seed)
    bench_ret = rng.normal(0.0005, 0.01, n)
    bench_ret[0] = 0.003  # return into the first day: outside the strategy period
    strat_ret = k * bench_ret
    strat_ret[0] = 0.0
    capital_list = (capital * np.cumprod(1 + strat_ret)).tolist()
    dates = pd.bdate_range('2022-01-03', periods=n).strftime('%Y%m%d')
    df_index = pd.DataFrame({
        'trade_date': dates,
        'pct_chg': bench_ret * 100,  # TuShare reports percent
    })
    return capital_list, df_index, bench_ret


class TestAlignReturns:
    def test_units_are_fractions_and_first_row_dropped(self):
        capital_list, df_index, bench_ret = _synthetic_pair(n=10)
        strat, bench = align_returns(pd.Series(capital_list), df_index)
        assert len(strat) == len(bench) == 9
        np.testing.assert_allclose(bench.to_numpy(), bench_ret[1:])
        np.testing.assert_allclose(strat.to_numpy(), 1.5 * bench_ret[1:])

    def test_does_not_modify_benchmark(self):
        capital_list, df_index, _ = _synthetic_pair(n=10)
        before = df_index.copy()
        align_returns(pd.Series(capital_list), df_index)
        pd.testing.assert_frame_equal(df_index, before)

    def test_length_mismatch_raises(self):
        capital_list, df_index, _ = _synthetic_pair(n=10)
        with pytest.raises(ValueError, match='length'):
            align_returns(pd.Series(capital_list[1:]), df_index)

    def test_descending_benchmark_raises(self):
        capital_list, df_index, _ = _synthetic_pair(n=10)
        with pytest.raises(ValueError, match='ascending'):
            align_returns(pd.Series(capital_list), df_index.iloc[::-1])

    def test_align_by_date_with_longer_benchmark(self):
        # the back tester drops the moving-average warm-up rows from stock
        # data, so the benchmark covers more days than the capital list
        capital_list, df_index, bench_ret = _synthetic_pair(n=30)
        extra = pd.DataFrame({
            'trade_date': pd.bdate_range('2021-12-01', periods=5).strftime('%Y%m%d'),
            'pct_chg': [9.0, -9.0, 9.0, -9.0, 9.0],
        })
        df_long = pd.concat([extra, df_index], ignore_index=True)
        strat, bench = align_returns(pd.Series(capital_list), df_long,
                                     capital_dates=df_index['trade_date'])
        np.testing.assert_allclose(bench.to_numpy(), bench_ret[1:])
        np.testing.assert_allclose(strat.to_numpy(), 1.5 * bench_ret[1:])

    def test_align_by_date_missing_date_raises(self):
        capital_list, df_index, _ = _synthetic_pair(n=10)
        with pytest.raises(ValueError, match='no data'):
            align_returns(pd.Series(capital_list), df_index.drop(index=4),
                          capital_dates=df_index['trade_date'])

    def test_align_by_date_unsorted_dates_raises(self):
        capital_list, df_index, _ = _synthetic_pair(n=10)
        with pytest.raises(ValueError, match='ascending'):
            align_returns(pd.Series(capital_list), df_index,
                          capital_dates=df_index['trade_date'][::-1])


class TestCalBeta:
    def test_scaled_returns(self):
        rng = np.random.default_rng(1)
        bench = pd.Series(rng.normal(0, 0.01, 100))
        assert cal_beta(1.5 * bench, bench) == pytest.approx(1.5)
        assert cal_beta(-0.5 * bench, bench) == pytest.approx(-0.5)

    def test_beta_with_idiosyncratic_noise(self):
        rng = np.random.default_rng(2)
        bench = pd.Series(rng.normal(0, 0.01, 20_000))
        noise = pd.Series(rng.normal(0, 0.005, 20_000))
        assert cal_beta(1.5 * bench + noise, bench) == pytest.approx(1.5, abs=0.02)

    def test_zero_benchmark_variance_is_nan(self):
        bench = pd.Series(np.zeros(10))
        strat = pd.Series(np.linspace(-0.01, 0.01, 10))
        assert np.isnan(cal_beta(strat, bench))


class TestAnnualizeReturn:
    def test_one_year_is_unchanged(self):
        assert annualize_return(0.1, 250) == pytest.approx(0.1)

    def test_two_years(self):
        # 21% over two years is 10% a year
        assert annualize_return(0.21, 500) == pytest.approx(0.1)

    def test_half_year(self):
        assert annualize_return(0.05, 125) == pytest.approx(1.05 ** 2 - 1)

    def test_undefined_cases_are_nan(self):
        assert np.isnan(annualize_return(0.1, 0))
        assert np.isnan(annualize_return(-1.0, 250))


class TestCalRiskIndicatorBeta:
    def test_beta_is_1_5_and_alpha_consistent(self, tmp_path):
        capital_list, df_index, bench_ret = _synthetic_pair()
        before = df_index.copy()
        df_trade = pd.DataFrame({'capital': capital_list[::50]})

        df_result = cal_risk_indicator(100_000.0, 0.04, capital_list, df_trade,
                                       df_index, str(tmp_path))

        beta = df_result.loc['beta', 'value']
        assert beta == pytest.approx(1.5)

        # 250 capital points -> 249 daily returns; annualize both legs
        n = len(capital_list) - 1
        rise = (capital_list[-1] / capital_list[0]) ** (250 / n) - 1
        sign_rise = np.prod(1 + bench_ret[1:]) ** (250 / n) - 1
        expected_alpha = rise - (0.04 + 1.5 * (sign_rise - 0.04))
        assert df_result.loc['alpha', 'value'] == pytest.approx(expected_alpha)

        pd.testing.assert_frame_equal(df_index, before)
        assert (tmp_path / 'risk_indicator.csv').exists()

    def test_beta_with_capital_dates(self, tmp_path):
        capital_list, df_index, _ = _synthetic_pair()
        df_trade = pd.DataFrame({'capital': capital_list[::50]})
        df_result = cal_risk_indicator(100_000.0, 0.04, capital_list[20:],
                                       df_trade, df_index, str(tmp_path),
                                       capital_dates=df_index['trade_date'][20:])
        assert df_result.loc['beta', 'value'] == pytest.approx(1.5)

    def test_alpha_uses_annual_rates_for_multi_year_backtest(self, tmp_path):
        # Two years where the strategy earns exactly the risk-free rate with
        # no market exposure (benchmark moves, strategy return is constant).
        n_days = 500
        rf = 0.04
        daily_rf = (1 + rf) ** (1 / 250) - 1
        rng = np.random.default_rng(3)
        bench_ret = rng.normal(0.0, 0.01, n_days + 1)
        capital_list = (100_000.0 * (1 + daily_rf) ** np.arange(n_days + 1)).tolist()
        df_index = pd.DataFrame({
            'trade_date': pd.bdate_range('2020-01-01', periods=n_days + 1).strftime('%Y%m%d'),
            'pct_chg': bench_ret * 100,
        })
        df_trade = pd.DataFrame({'capital': capital_list[::100]})

        df_result = cal_risk_indicator(100_000.0, rf, capital_list, df_trade,
                                       df_index, str(tmp_path))

        assert df_result.loc['beta', 'value'] == pytest.approx(0.0, abs=1e-9)
        # earning the risk-free rate with beta 0 is zero alpha; comparing the
        # two-year total (~8.16%) with the yearly 4% would give ~4.16%
        assert df_result.loc['alpha', 'value'] == pytest.approx(0.0, abs=1e-9)
