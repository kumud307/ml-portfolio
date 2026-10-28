"""Simulate the 'top N stocks for the next H days' portfolio, after costs and tax.

Every H trading days we buy the N highest-scoring stocks (equal amounts, at
most MAX_PER_SECTOR per sector), hold them for H days, then repeat. Trading
costs are charged on the part of the portfolio that changes. Each strategy is
just a different way of scoring stocks, so they are all tested identically.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (COST_ROUND_TRIP, HOLD_BUFFER, MAX_PER_SECTOR, RISK_FREE_RATE,
                     SHORT_TERM_TAX, TOP_N)
from .features import target_columns

ML = "ML model"
MOMENTUM = "Momentum (12-1 month)"
MOMENTUM_3M = "Momentum (3 months)"
BLEND = "ML + Momentum"
REVERSAL = "Reversal (worst week)"
AVERAGE = "Average stock"
INDEX = "NIFTY 500"

STRATEGIES = [ML, BLEND, MOMENTUM, MOMENTUM_3M, REVERSAL]
BENCHMARKS = [AVERAGE, INDEX]

STRATEGY_NOTES = {
    ML: "Gradient-boosting model's predicted rank",
    BLEND: "Average of the ML rank and the 12-1 momentum rank",
    MOMENTUM: "Best performers over the last year, skipping the latest month",
    MOMENTUM_3M: "Best performers vs the market over the last 3 months",
    REVERSAL: "Biggest losers of the last week (bet on a bounce)",
    AVERAGE: "Equal amount in every eligible stock",
    INDEX: "Simply holding the NIFTY 500 index",
}


def strategy_scores(frame):
    """One score column per strategy (higher = buy first).

    Works on a multi-date panel (ranks are taken within each date) or on a
    single day's table indexed by symbol.
    """
    def pct_rank(column):
        if isinstance(frame.index, pd.MultiIndex):
            return frame.groupby(level="date")[column].rank(pct=True)
        return frame[column].rank(pct=True)

    return pd.DataFrame({
        ML: frame["ml_score"],
        BLEND: (pct_rank("ml_score") + pct_rank("mom_12_1")) / 2,
        MOMENTUM: frame["mom_12_1"],
        MOMENTUM_3M: frame["rel_60"],
        REVERSAL: -frame["ret_5"],
    }, index=frame.index)


def select_top(scores, sectors, n=TOP_N, max_per_sector=MAX_PER_SECTOR,
               held=(), buffer=HOLD_BUFFER):
    """Best `n` symbols by score, at most `max_per_sector` from any sector.

    Stocks already `held` are kept while they still rank in the top `buffer`.
    Selling a stock that slipped from #8 to #12 and buying #10 instead costs
    money for almost no expected gain.
    """
    ranked = list(scores.dropna().sort_values(ascending=False).index)
    keep = [s for s in ranked[:buffer] if s in set(held)]
    picks, per_sector = [], {}
    for symbol in keep + ranked:
        sector = sectors[symbol]
        if symbol not in picks and per_sector.get(sector, 0) < max_per_sector:
            picks.append(symbol)
            per_sector[sector] = per_sector.get(sector, 0) + 1
            if len(picks) == n:
                break
    return picks


@dataclass
class BacktestResult:
    periods: pd.DataFrame    # one row per (holding period, strategy): returns, turnover
    summary: pd.DataFrame    # one row per strategy: CAGR, Sharpe, drawdown, IC, ...
    equity: pd.DataFrame     # growth of 1 rupee, one column per strategy
    horizon: int
    top_n: int

    @property
    def start(self):
        return self.equity.index[0]

    @property
    def end(self):
        return self.equity.index[-1]

    @property
    def best_strategy(self):
        """Highest after-cost Sharpe ratio among the stock-picking strategies."""
        return self.summary.loc[STRATEGIES, "sharpe"].idxmax()


def _max_drawdown(equity):
    return (equity / equity.cummax() - 1).min()


def _after_tax_cagr(net, dates, periods_per_year):
    """Tax each calendar year's profit at the short-term rate (losses untaxed)."""
    yearly = (1 + net).groupby(pd.DatetimeIndex(dates).year).prod() - 1
    taxed = yearly.where(yearly <= 0, yearly * (1 - SHORT_TERM_TAX))
    years = len(net) / periods_per_year
    return (1 + taxed).prod() ** (1 / years) - 1


def _information_coefficient(panel, scores):
    """Per-date rank correlation between score and the actual excess return."""
    joined = pd.concat([scores.rename("score"), panel["fwd_excess"]], axis=1).dropna()
    by_date = joined.groupby(level="date")
    ic = by_date.apply(lambda day: day["score"].corr(day["fwd_excess"], method="spearman"))
    t_stat = ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan
    return ic.mean(), t_stat


def run_backtest(panel, ml_score, horizon, top_n=TOP_N, buffer=HOLD_BUFFER):
    """`ml_score` holds out-of-sample model predictions for the rows to test.

    `buffer` should be small compared with the number of stocks (30 of ~480
    here); if it covers most of the market, nothing is ever sold.
    """
    return_col, excess_col, _ = target_columns(horizon)
    tested = panel[["sector", "mom_12_1", "rel_60", "ret_5", return_col, excess_col]].rename(
        columns={return_col: "fwd_return", excess_col: "fwd_excess"})
    tested["ml_score"] = ml_score
    tested = tested.dropna(subset=["ml_score", "fwd_return"])
    all_dates = np.sort(tested.index.get_level_values("date").unique())
    rebalance = all_dates[::horizon]               # non-overlapping holding periods
    tested = tested[tested.index.get_level_values("date").isin(rebalance)]
    scores = strategy_scores(tested)

    rows = []
    previous = {name: set() for name in STRATEGIES}
    for date, day in tested.groupby(level="date"):
        day = day.droplevel("date")
        day_scores = scores.xs(date, level="date")
        index_return = (day["fwd_return"] - day["fwd_excess"]).mean()
        rows.append({"date": date, "strategy": AVERAGE, "gross": day["fwd_return"].mean(),
                     "turnover": 0.0, "index_return": index_return})
        rows.append({"date": date, "strategy": INDEX, "gross": index_return,
                     "turnover": 0.0, "index_return": index_return})
        for name in STRATEGIES:
            picks = select_top(day_scores[name], day["sector"], top_n,
                               held=previous[name], buffer=buffer)
            if not picks:
                continue
            held = set(picks)
            turnover = 1 - len(held & previous[name]) / len(held)
            previous[name] = held
            rows.append({"date": date, "strategy": name, "gross": day.loc[picks, "fwd_return"].mean(),
                         "turnover": turnover, "index_return": index_return})

    periods = pd.DataFrame(rows)
    periods["net"] = periods["gross"] - COST_ROUND_TRIP * periods["turnover"]
    periods["excess"] = periods["net"] - periods["index_return"]

    per_year = 252 / horizon
    equity, summary = {}, {}
    for name, group in periods.groupby("strategy", sort=False):
        net = group.set_index("date")["net"]
        curve = (1 + net).cumprod()
        equity[name] = curve
        cagr = curve.iloc[-1] ** (per_year / len(net)) - 1
        volatility = net.std() * np.sqrt(per_year)
        ic, ic_t = (_information_coefficient(tested, scores[name])
                    if name in STRATEGIES else (np.nan, np.nan))
        summary[name] = {
            "avg_period_return": net.mean(),
            "avg_excess": group["excess"].mean(),
            "hit_rate": (group["excess"] > 0).mean(),
            "cagr": cagr,
            "cagr_after_tax": (_after_tax_cagr(net, net.index, per_year)
                               if name in STRATEGIES else cagr),
            "volatility": volatility,
            "sharpe": (net.mean() * per_year - RISK_FREE_RATE) / volatility,
            "max_drawdown": _max_drawdown(curve),
            "worst_period": net.min(),
            "turnover": group["turnover"].mean(),
            "ic": ic,
            "ic_t_stat": ic_t,
        }

    order = STRATEGIES + BENCHMARKS
    summary = pd.DataFrame(summary).T.reindex(order)
    return BacktestResult(periods, summary, pd.DataFrame(equity)[order], horizon, top_n)
