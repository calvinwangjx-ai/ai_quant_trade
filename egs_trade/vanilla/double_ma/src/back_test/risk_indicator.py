# -*- coding: utf-8 -*-
# @Author   : liyi
# @Time     : 2022/7/8 8:19
# @File     : risk_indicator.py
# @Project  : ai_quant_trade
# Copyright (c) Personal 2022 liyi
# Function Description:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, Sequence, Tuple

import ffn
import numpy as np
import pandas as pd

from src.tools.log.log_util import addlog, log

# trading days in one year, used to annualize returns
TRADING_DAYS_PER_YEAR = 250


def align_returns(capital_series: pd.Series,
                  df_index: pd.DataFrame,
                  capital_dates: Optional[Sequence] = None
                  ) -> Tuple[pd.Series, pd.Series]:
    """
    Build strategy and benchmark daily returns on the same dates and units.

    Both returned series are fractional returns (0.01 == 1%). TuShare's
    ``pct_chg`` is in percent, so the benchmark is divided by 100 here.
    ``df_index`` is not modified.

    The first capital point has no previous value, so its return is undefined;
    that row is dropped from both series rather than filled with 0, because a
    0 strategy return paired with a non-zero benchmark return would bias beta.

    :param capital_series: account capital, one value per trading day, in
        chronological order
    :param df_index: benchmark data with a ``pct_chg`` column in percent and,
        optionally, a ``trade_date`` column
    :param capital_dates: trade date of each capital value. If given, the
        benchmark is matched by ``trade_date``; otherwise the two are matched
        by position and must have the same length.
    :return: (strategy returns, benchmark returns), equal length, 0-based index
    """
    capital_series = pd.Series(capital_series, dtype=float).reset_index(drop=True)
    bench_pct = df_index['pct_chg'].astype(float) / 100.0

    if capital_dates is not None:
        if 'trade_date' not in df_index.columns:
            raise ValueError('capital_dates given but benchmark has no trade_date column')
        dates = pd.Series(list(capital_dates)).astype(str)
        if len(dates) != len(capital_series):
            raise ValueError('capital_dates length %d != capital length %d'
                             % (len(dates), len(capital_series)))
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError('capital_dates must be unique and in ascending order')

        bench = pd.Series(bench_pct.to_numpy(),
                          index=df_index['trade_date'].astype(str).to_numpy())
        if bench.index.duplicated().any():
            raise ValueError('benchmark trade_date contains duplicates')
        missing = dates[~dates.isin(bench.index)]
        if len(missing):
            raise ValueError('benchmark has no data for %d trading date(s), e.g. %s'
                             % (len(missing), missing.iloc[0]))
        bench_ret = pd.Series(bench.reindex(dates.to_numpy()).to_numpy())
    else:
        if len(df_index) != len(capital_series):
            raise ValueError('benchmark length %d != capital length %d; pass '
                             'capital_dates to align by trade_date'
                             % (len(df_index), len(capital_series)))
        if 'trade_date' in df_index.columns:
            bench_dates = df_index['trade_date'].astype(str)
            if bench_dates.duplicated().any() or not bench_dates.is_monotonic_increasing:
                raise ValueError('benchmark trade_date must be unique and in ascending order')
        bench_ret = bench_pct.reset_index(drop=True)

    strategy_ret = capital_series.pct_change()
    return (strategy_ret.iloc[1:].reset_index(drop=True),
            bench_ret.iloc[1:].reset_index(drop=True))


def cal_beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    Beta = Cov(Dp, Dm) / Var(Dm), both with ddof=1.

    Returns NaN when the benchmark variance is zero or undefined.
    """
    var = bench_ret.var()
    if not np.isfinite(var) or var == 0:
        return float('nan')
    return float(strategy_ret.cov(bench_ret) / var)


def annualize_return(total_return: float, n_periods: int,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Convert a compounded return over ``n_periods`` daily periods to a yearly
    rate: (1 + R) ^ (periods_per_year / n_periods) - 1.

    Returns NaN when there are no periods or the capital was wiped out
    (1 + R <= 0), where a yearly rate is undefined.
    """
    growth = 1.0 + float(total_return)
    if n_periods <= 0 or growth <= 0:
        return float('nan')
    return growth ** (periods_per_year / n_periods) - 1.0


@addlog(name='cal_risk_indicator')
def cal_risk_indicator(capital: float,
                       base_rise: float,
                       capital_list: list,
                       df_trade: pd.DataFrame,
                       df_index: pd.DataFrame,
                       metrics_save_path: str,
                       capital_dates: Optional[Sequence] = None) -> pd.DataFrame:
    """
    :param capital: initial money you have for stock trading
    :param base_rise: e.g. Bank Deposit Rate, risk-free interest rate
    :param capital_list: capital changing list during trading period
    :param df_trade: detailed trading info dataframe
    :param df_index: benchmark of market index (``pct_chg`` in percent)
    :param metrics_save_path: path to save risk indicator
    :param capital_dates: trade date of each value in ``capital_list``; used to
        align the benchmark by ``trade_date``. Without it, ``df_index`` must
        have one row per capital value, in the same order.
    :return: dataframe with all calculated risk indicators
    """
    indicator_dict = {}
    capital_series = pd.Series(capital_list)

    # 1. calculate simple rate of return
    # represent capital changing (Pend - Pstart) / Pstart * 100%
    capital_returns = ffn.to_returns(capital_series)
    # calculate total rate of return
    total_returns = ffn.calc_total_return(capital_series)
    indicator_dict['total_returns'] = total_returns
    log.info('Total Return Rate: %.5f' % total_returns)

    # 2. Total Annal Return
    # ((1+P)^(250/n) - 1) * 100%, P is capital returns
    # different from annual return if trading time less than a year
    # due to its not the real annual return, it's a converted value
    # 250 is the trading days in one year
    annal_returns = ffn.annualize(total_returns,
                                  len(capital_list), one_year=TRADING_DAYS_PER_YEAR)
    indicator_dict['annal_returns'] = annal_returns
    log.info('Annual Return Rate: %.5f' % annal_returns)

    # 3. Beta
    # Use to evaluate correlation between returns of strategy and benchmark returns
    #  we usually use Shanghai securities composite index as benchmark
    #  if benchmark return raise 1%, and strategy return raise 1.5%,
    #  and beta is 1.5, vice versa -1.5
    #  if market place is good, the larger value the better
    #  if not good, the smaller the better
    #  it is usually explained as system risk
    #  Beta = Cov(Dp, Dm) / Var(Dm)
    #  Cov: covariance, Var: variance of benchmark return
    # Beta > 1: strategy changing larger than benchmark one, vice versa
    # todo: due to index data not available from tushare api without pay
    sign_rise = rise = beta = None
    if len(df_index):
        # both series are fractional returns on the same dates
        strategy_pct_chg, bench_pct_chg = align_returns(capital_series,
                                                        df_index,
                                                        capital_dates)
        beta = cal_beta(strategy_pct_chg, bench_pct_chg)
        indicator_dict['beta'] = beta
        log.info('Beta Value: %.5f' % beta)

        # compounded return of benchmark and strategy over the same days,
        #  annualized so they match base_rise, which is a yearly rate
        n_periods = len(strategy_pct_chg)
        sign_rise = annualize_return((bench_pct_chg + 1).prod() - 1, n_periods)
        rise = annualize_return((strategy_pct_chg + 1).prod() - 1, n_periods)

    # 4. Alpha
    # alpha is non-system risk. It represents returns which has no
    #  relationship with market fluctuation. E.g., if strategy return
    #  is 20%, benchmark return is 10%, the alpha is 10%.
    #  Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    #  Rp: strategy annual returns, Rm: benchmark annual returns,
    #    both annualized from the backtest period (see step 3)
    #  Beta: bata ratio is the one calculated above in 3
    #  Rf is risk-free interest rate, can be easily thought as
    #    Bank Deposit Rate, usually set as 0.04
    # todo: due to index data not available from tushare api without pay
    if len(df_index):
        alpha = rise - (base_rise + beta * (sign_rise - base_rise))
        indicator_dict['alpha'] = alpha
        log.info('Alpha Value: %.5f' % alpha)

    # 5. calculate sharp rate
    # used to evaluate every one risk u take, the non-risk returns
    # the larger value is, the more returns while more risk u take, vice versa
    # usually, the larger value is, the better
    # Sharp Ratio = (Rp - Rf) / Sigma(p)
    # Rp: annual return, Rf: no-risk interest rate,
    # Sigma(p): changing rate of the strategy (Annual standard deviation of  Strategy Returns)
    sharp_ratio = ffn.calc_sharpe(capital_returns, nperiods=TRADING_DAYS_PER_YEAR)
    indicator_dict['sharp_ratio'] = sharp_ratio
    log.info('Sharp Rate: %.5f' % sharp_ratio)

    # 6. calculate max withdraw
    # evaluate max loss of the strategy, the smaller the value the better
    # max_draw_down = (Px-Py)/Px, (Px, PY: max and min value during a time)
    max_withdraw = ffn.calc_max_drawdown(capital_series)
    indicator_dict['max_withdraw'] = max_withdraw
    log.info('Max Withdraw: %.5f' % max_withdraw)

    # 7. Sortino Ratio
    # evaluate strategy loss, the larger value the better
    # Sortino Ratio = (Rp - Rf) / Sigma(d)
    # Rp: annual return, Rf: no-risk interest rate, Sigma(d): strategy downward volatility
    sortino_ratio = ffn.calc_sortino_ratio(capital_returns, nperiods=TRADING_DAYS_PER_YEAR)
    indicator_dict['sortino_ratio'] = sortino_ratio
    log.info('Sortino Rate: %.5f' % sortino_ratio)

    # 8. win rate
    # e.g. 8 wins in 10 trades, then win rate is 80%
    # loop through each trade and neglect first trade
    #  (? whether include first trade)
    # type1: Capital Now > Capital Pre, count 1 win
    # type2: Capital Now > Original Capital, count 1 win
    # type3: type 1 and 2 all satisfied
    # todo: need to confirm it's correct (I implemented type3)
    # fixme: if u have multiple trades, it had to distinguish which trade wins or lose,
    #  it might result wrong calculation
    total_trades = len(df_trade)  # todo: whether to remove 1 ?
    win_cnt = 0
    # neglect 1st trade, due to no win or loss for 1st buy
    for i in range(1, total_trades):
        if df_trade['capital'].iloc[i] > df_trade['capital'].iloc[i - 1]:
            if df_trade['capital'].iloc[i] > capital:
                win_cnt += 1
    win_rate = win_cnt / total_trades
    indicator_dict['win_rate'] = win_rate
    log.info('Win Rate: %.5f' % win_rate)

    # 9. profit-loss ratio
    # during a time period of trading, if profit is 12000,
    # loss is 8000, then the ratio is 1.5
    # loop through each trade and neglect first trade
    # type1: sum of profit / sum of loss
    # type2: max of profit / max of loss
    # todo: need to confirm it's correct (I implemented type2)
    # fixme: repeat calculation with step 8
    #   to make it clear, I split to write
    max_profit, max_loss = 0, 1.0e-10  # 1.0e-10 to avoid divide by zero
    # neglect 1st trade, due to no win or loss for 1st buy
    for i in range(1, total_trades):
        if df_trade['capital'].iloc[i] >= capital:
            max_profit = max(max_profit, df_trade['capital'].iloc[i] - capital)
        else:
            max_loss = max(max_loss, capital - df_trade['capital'].iloc[i])
    profit_loss_ratio = max_profit / max_loss
    indicator_dict['profit_loss_ratio'] = profit_loss_ratio
    log.info('Profit-Loss Ratio: %.5f' % profit_loss_ratio)

    # 10. Max Consecutive Losses (todo)
    max_con_losses, cur_con_losses, pre_loss = 0, 0, 0
    pre_flag = False
    # neglect 1st trade, due to no win or loss for 1st buy
    for i in range(1, total_trades):
        if df_trade['capital'].iloc[i] < df_trade['capital'].iloc[i-1]:
            loss = df_trade['capital'].iloc[i-1] - df_trade['capital'].iloc[i]
            if pre_flag:
                cur_con_losses += loss + pre_loss
                max_con_losses = max(max_con_losses, cur_con_losses)
                pre_loss = 0
            else:
                max_con_losses = max(max_con_losses, loss)
                cur_con_losses = 0
                pre_loss = loss

            pre_flag = True
            continue

    # for i in range(1, total_trades):
    #     if df_trade['capital'].iloc[i] < capital:
    #         if df_trade['capital'].iloc[i] < df_trade['capital'].iloc[i-1]:
    #             loss = capital - df_trade['capital'].iloc[i]
    #             if pre_flag:
    #                 cur_con_losses += loss + pre_loss
    #                 max_con_losses = max(max_con_losses, cur_con_losses)
    #                 pre_loss = 0
    #             else:
    #                 max_con_losses = max(max_con_losses, loss)
    #                 cur_con_losses = 0
    #                 pre_loss = loss
    #
    #             pre_flag = True
    #             continue

        pre_flag = False
        cur_con_losses = 0
        pre_loss = 0

    indicator_dict['max_con_losses'] = max_con_losses
    log.info('Max Consecutive Losses: %.5f' % max_con_losses)

    # === save to the file ===
    # 1. save risk indicator
    # 用单列 DataFrame 构造，避免 pandas 3.x 对不均匀形状 values() 的 ValueError
    df_result = pd.DataFrame({'value': list(indicator_dict.values())},
                             index=list(indicator_dict.keys()))
    save_path = os.path.join(metrics_save_path, 'risk_indicator.csv')
    df_result.to_csv(save_path)

    # 2. save trading info
    save_path = os.path.join(metrics_save_path, 'trading_info.csv')
    df_trade.to_csv(save_path)

    return df_result
