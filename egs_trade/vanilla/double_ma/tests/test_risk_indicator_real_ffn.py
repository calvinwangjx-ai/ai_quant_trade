# -*- coding: utf-8 -*-
"""Run ``cal_risk_indicator`` against the real ``ffn`` package.

``tests/test_all.py`` mocks ``ffn`` via ``sys.modules.setdefault``, which
hid a crash in ffn >= 1.x: ``calc_sharpe`` / ``calc_sortino_ratio`` raise
``ValueError: Cannot infer return frequency`` for an integer-indexed
series unless ``nperiods`` is given. This test exercises the real library.
"""
import importlib
import math
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def _import_real_ffn():
    """Import the real ffn even if another test already put a mock in
    ``sys.modules``. Skip when ffn isn't installed."""
    cached = sys.modules.get('ffn')
    if isinstance(cached, mock.Mock):
        del sys.modules['ffn']
    try:
        return importlib.import_module('ffn')
    except ImportError:
        if cached is not None:
            sys.modules['ffn'] = cached
        pytest.skip('ffn is not installed')


@pytest.fixture
def real_ffn_risk_module():
    cached = sys.modules.get('ffn')
    real_ffn = _import_real_ffn()
    import src.back_test.risk_indicator as risk_indicator
    # Make sure the module under test uses the real ffn, not a mock that
    # may have been bound at import time by another test.
    with mock.patch.object(risk_indicator, 'ffn', real_ffn):
        yield risk_indicator
    if cached is not None:
        sys.modules['ffn'] = cached


def test_cal_risk_indicator_with_real_ffn(real_ffn_risk_module, tmp_path):
    capital = 100_000.0
    capital_list = [100_000, 100_500, 99_800, 101_200, 100_900,
                    102_300, 101_700, 103_000, 102_400, 104_100]
    df_trade = pd.DataFrame({
        'time_index': range(4),
        'capital': [100_000, 101_200, 100_900, 104_100],
        'trade_detail': ['buy', 'sell', 'buy', 'sell'],
    })
    df_index = pd.DataFrame()  # no benchmark: skip beta / alpha

    df_result = real_ffn_risk_module.cal_risk_indicator(
        capital, 0.04, capital_list, df_trade, df_index, str(tmp_path))

    assert (tmp_path / 'risk_indicator.csv').exists()
    assert (tmp_path / 'trading_info.csv').exists()

    values = df_result['value']
    for key in ('total_returns', 'annal_returns', 'sharp_ratio',
                'max_withdraw', 'sortino_ratio'):
        assert key in values.index
        assert math.isfinite(float(values[key])), key

    assert values['total_returns'] == pytest.approx(0.041)
    assert values['max_withdraw'] < 0
    assert values['sharp_ratio'] > 0
    assert values['sortino_ratio'] > 0
