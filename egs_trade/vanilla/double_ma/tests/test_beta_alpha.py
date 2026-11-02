# -*- coding: utf-8 -*-
"""Unit tests for the beta / alpha helpers in risk_indicator."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# The helpers are pure pandas; ffn is only needed by the module import.
try:
    import ffn  # noqa: F401
except ImportError:
    sys.modules.setdefault('ffn', mock.MagicMock())

from src.back_test.risk_indicator import (  # noqa: E402
    TRADING_DAYS_PER_YEAR, align_daily_returns, calc_beta_alpha)

RF = 0.04


def _benchmark_returns(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def _annualized(ret):
    return (1 + ret).prod() ** (TRADING_DAYS_PER_YEAR / len(ret)) - 1


class TestCalcBetaAlpha:
    def test_same_as_benchmark(self):
        rm = _benchmark_returns()
        beta, alpha = calc_beta_alpha(rm, rm, RF)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.0, abs=1e-12)

    def test_leveraged_benchmark(self):
        rm = _benchmark_returns()
        rs = 1.5 * rm
        beta, alpha = calc_beta_alpha(rs, rm, RF)
        assert beta == pytest.approx(1.5)
        expected = _annualized(rs) - (RF + 1.5 * (_annualized(rm) - RF))
        assert alpha == pytest.approx(expected)

    def test_uncorrelated_constant_return(self):
        rm = _benchmark_returns()
        rs = pd.Series(np.full(len(rm), 0.001))
        beta, alpha = calc_beta_alpha(rs, rm, RF)
        assert beta == pytest.approx(0.0, abs=1e-12)
        assert alpha == pytest.approx(_annualized(rs) - RF)

    def test_flat_benchmark_is_undefined(self):
        rm = pd.Series(np.zeros(50))
        rs = _benchmark_returns(50)
        beta, alpha = calc_beta_alpha(rs, rm, RF)
        assert np.isnan(beta) and np.isnan(alpha)

    def test_too_few_days_is_undefined(self):
        beta, alpha = calc_beta_alpha(pd.Series([0.01]), pd.Series([0.02]), RF)
        assert np.isnan(beta) and np.isnan(alpha)


class TestAlignDailyReturns:
    def test_aligns_by_trade_date_and_converts_percent(self):
        # Benchmark covers a longer period (the moving-average warm-up days
        # the strategy skips) and has int trade dates, as a TuShare csv read
        # back with pandas does; capital dates are strings.
        dates = pd.date_range('2022-01-01', periods=30, freq='D').strftime('%Y%m%d')
        pct_chg = np.round(np.random.default_rng(1).normal(0, 1, 30), 3)
        df_index = pd.DataFrame({'trade_date': dates.astype(int),
                                 'pct_chg': pct_chg})
        df_before = df_index.copy()

        cap_dates = list(dates[10:])          # strategy starts on day 10
        bench = pd.Series(pct_chg[10:] / 100)
        capital = 100_000 * (1 + 2 * bench).cumprod()
        rs, rm = align_daily_returns(pd.Series(capital.values), df_index,
                                     cap_dates)

        assert len(rs) == len(rm) == len(cap_dates) - 1  # first pct is NaN
        np.testing.assert_allclose(rm.values, pct_chg[11:] / 100)
        np.testing.assert_allclose(rs.values, 2 * rm.values)
        beta, _ = calc_beta_alpha(rs, rm, RF)
        assert beta == pytest.approx(2.0)
        # caller's benchmark frame must not be modified
        pd.testing.assert_frame_equal(df_index, df_before)

    def test_positional_when_lengths_match(self):
        df_index = pd.DataFrame({'trade_date': range(5),
                                 'pct_chg': [0.0, 1.0, -1.0, 2.0, 0.5]})
        capital = pd.Series([100.0, 101.0, 99.99, 101.9898, 102.499749])
        rs, rm = align_daily_returns(capital, df_index)
        assert len(rs) == 4
        np.testing.assert_allclose(rs.values, rm.values, rtol=1e-6)

    def test_mismatched_lengths_without_dates_gives_empty(self):
        df_index = pd.DataFrame({'trade_date': range(5),
                                 'pct_chg': np.ones(5)})
        rs, rm = align_daily_returns(pd.Series(np.arange(1.0, 4.0)), df_index)
        assert rs.empty and rm.empty

    def test_wrong_number_of_dates_raises(self):
        df_index = pd.DataFrame({'trade_date': range(5),
                                 'pct_chg': np.ones(5)})
        with pytest.raises(ValueError):
            align_daily_returns(pd.Series(np.arange(1.0, 4.0)), df_index,
                                capital_dates=[1, 2])
