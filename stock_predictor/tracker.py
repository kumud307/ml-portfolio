"""Live track record: save each day's picks, then score them once their
holding period has passed.

A backtest can be tuned (consciously or not) until it looks good. Picks
written down *before* the outcome is known can't, so a growing live record
is the most convincing evidence a strategy works.
"""

import pandas as pd

from .config import EXECUTION_LAG_DAYS
from .paths import ROOT
from .universe import BENCHMARK

TRACK_FILE = ROOT / "track_record" / "picks.csv"
COLUMNS = ["as_of", "horizon", "strategy", "pick", "symbol", "name", "sector", "signal_price"]


def load_picks(path=TRACK_FILE):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["as_of"])


def record(as_of, horizon, strategy, picks, path=TRACK_FILE):
    """Append today's picks, once per (date, horizon, strategy). Returns True if saved."""
    existing = load_picks(path)
    already = ((existing.as_of == pd.Timestamp(as_of)) & (existing.horizon == horizon)
               & (existing.strategy == strategy)).any()
    if already or picks.empty:
        return False
    rows = pd.DataFrame({
        "as_of": pd.Timestamp(as_of),
        "horizon": horizon,
        "strategy": strategy,
        "pick": range(1, len(picks) + 1),
        "symbol": picks.index,
        "name": picks["name"].to_numpy(),
        "sector": picks["sector"].to_numpy(),
        "signal_price": picks["price"].to_numpy(),
    })
    path.parent.mkdir(exist_ok=True)
    pd.concat([existing, rows]).to_csv(path, index=False)
    return True


def evaluate(close, path=TRACK_FILE):
    """One row per recorded batch: how its picks have done vs the NIFTY 500.

    Like the backtest, each batch is bought at the close EXECUTION_LAG_DAYS
    after the signal and held `horizon` trading days. Batches still inside
    their holding period show the return so far.
    """
    picks = load_picks(path)
    if picks.empty:
        return pd.DataFrame()
    dates = close.index
    rows = []
    for (as_of, horizon, strategy), batch in picks.groupby(["as_of", "horizon", "strategy"]):
        start = dates.searchsorted(as_of) + EXECUTION_LAG_DAYS
        if start >= len(dates):
            rows.append({"as_of": as_of, "horizon": horizon, "strategy": strategy,
                         "status": "Waiting to buy", "stocks": len(batch)})
            continue
        end = min(start + horizon, len(dates) - 1)
        held = [s for s in batch.symbol if s in close.columns]
        stock_returns = close[held].iloc[end] / close[held].iloc[start] - 1
        index_return = close[BENCHMARK].iloc[end] / close[BENCHMARK].iloc[start] - 1
        rows.append({
            "as_of": as_of,
            "horizon": horizon,
            "strategy": strategy,
            "status": "Finished" if start + horizon <= len(dates) - 1 else "In progress",
            "bought": dates[start],
            "until": dates[end],
            "stocks": len(held),
            "portfolio_return": stock_returns.mean(),
            "nifty500_return": index_return,
            "vs_market": stock_returns.mean() - index_return,
            "beat_market": int((stock_returns > index_return).sum()),
        })
    return pd.DataFrame(rows).sort_values("as_of", ascending=False)
