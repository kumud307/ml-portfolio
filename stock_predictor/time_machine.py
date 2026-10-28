"""Go back to a past date, pick stocks using only what was known then, and
compare the picks with what actually happened afterwards.

One date is a single example and can be lucky or unlucky; run_many repeats
the test on several dates for a fairer picture.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import AVERAGE, INDEX, STRATEGIES, select_top, strategy_scores
from .config import EXECUTION_LAG_DAYS, TOP_N
from .features import FEATURES, target_columns
from .model import fit

MIN_TRAINING_DAYS = 250   # need about a year of known answers before a test date


@dataclass
class TimeTravel:
    as_of: pd.Timestamp       # the day we "travel back" to
    end: pd.Timestamp         # when the holding period ended
    horizon: int
    picks: pd.DataFrame       # every strategy's picks and what they actually returned
    summary: pd.DataFrame     # per strategy: average return, vs market, hits
    actual_best: pd.DataFrame # the stocks that really did best
    stocks: int               # how many stocks were eligible that day


def available_dates(panel, horizon):
    """Trading days we can test: enough history before, the answer known after."""
    _, _, rank_col = target_columns(horizon)
    known = panel[rank_col].notna().groupby(level="date").any()
    dates = known.index[known]
    first_possible = dates.min() + pd.tseries.offsets.BDay(MIN_TRAINING_DAYS + horizon)
    return dates[dates >= first_possible]


def travel(panel, close, as_of, horizon, top_n=TOP_N):
    return_col, excess_col, _ = target_columns(horizon)
    dates = available_dates(panel, horizon)
    if len(dates) == 0 or pd.Timestamp(as_of) < dates[0]:
        raise ValueError("Too early: the model needs about a year of history before this date.")
    if pd.Timestamp(as_of) > dates[-1]:
        raise ValueError(f"Too recent: the next {horizon} trading days haven't all happened yet.")
    as_of = dates[dates <= pd.Timestamp(as_of)][-1]   # snap to a trading day

    model = fit(panel, horizon, as_of=as_of)
    day = panel.xs(as_of, level="date").copy()
    day = day[day[return_col].notna()]
    day["ml_score"] = model.predict(day[FEATURES])
    day["actual_return"] = day[return_col]
    day["actual_rank"] = day["actual_return"].rank(ascending=False).astype(int)
    index_return = (day[return_col] - day[excess_col]).mean()
    scores = strategy_scores(day)

    picks, summary = [], {}
    for name in STRATEGIES:
        chosen = day.loc[select_top(scores[name], day["sector"], top_n)]
        picks.append(chosen.assign(strategy=name, pick=np.arange(1, len(chosen) + 1)))
        summary[name] = _scorecard(chosen, index_return, len(day))
    summary[AVERAGE] = {"avg_return": day["actual_return"].mean(),
                        "vs_market": day["actual_return"].mean() - index_return}
    summary[INDEX] = {"avg_return": index_return, "vs_market": 0.0}

    end_position = close.index.get_loc(as_of) + EXECUTION_LAG_DAYS + horizon
    return TimeTravel(
        as_of=as_of,
        end=close.index[min(end_position, len(close) - 1)],
        horizon=horizon,
        picks=pd.concat(picks).reset_index(),
        summary=pd.DataFrame(summary).T,
        actual_best=day.nsmallest(top_n, "actual_rank").reset_index(),
        stocks=len(day),
    )


def _scorecard(chosen, index_return, n_stocks):
    returns = chosen["actual_return"]
    return {
        "avg_return": returns.mean(),
        "vs_market": returns.mean() - index_return,
        "beat_market": int((returns > index_return).sum()),
        "in_actual_top_10pct": int((chosen["actual_rank"] <= n_stocks * 0.10).sum()),
        "in_actual_top_10": int((chosen["actual_rank"] <= TOP_N).sum()),
        "best_pick": returns.max(),
        "worst_pick": returns.min(),
    }


def run_many(panel, close, horizon, every_n_months=3, top_n=TOP_N):
    """Repeat the time-travel test at regular dates; one row per (date, strategy)."""
    dates = available_dates(panel, horizon)
    targets = pd.date_range(dates[0], dates[-1], freq=f"{every_n_months}MS")
    rows = []
    for target in targets:
        result = travel(panel, close, target, horizon, top_n)
        for name, row in result.summary.iterrows():
            rows.append({"date": result.as_of, "strategy": name, **row.to_dict()})
    return pd.DataFrame(rows)
