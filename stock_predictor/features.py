"""Turn daily prices into a (date, symbol) table of model features and targets.

Only stocks that were eligible on a date (see universe.eligible_mask) get a
row for that date. Every feature on date t uses prices up to t only; the
targets look forward from t + EXECUTION_LAG_DAYS.
"""

import numpy as np
import pandas as pd

from .config import EXECUTION_LAG_DAYS, INDEX_DIVIDEND_YIELD
from .universe import BENCHMARK, eligible_mask

# Per-stock signals. The model sees each one as a percentile rank among that
# day's eligible stocks, so values are comparable across calm and wild years.
STOCK_FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120", "mom_12_1",
    "rel_20", "rel_60", "sector_rel_20", "sector_ret_60",
    "vol_20", "vol_60", "sma50_gap", "sma200_gap",
    "rsi_14", "drawdown_252", "volume_trend", "liquidity",
]
# Whole-market conditions ("regime"), the same for every stock on a date.
MARKET_FEATURES = ["mkt_ret_20", "mkt_ret_60", "mkt_sma200_gap", "mkt_vol_20",
                   "vix", "vix_change_20", "breadth_50"]
FEATURES = [f"{name}_rank" for name in STOCK_FEATURES] + MARKET_FEATURES


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def _sector_mean(frame, sector_of):
    """Each cell replaced by the average of its sector on that day."""
    groups = pd.Series(sector_of).reindex(frame.columns).fillna("Other")
    return frame.T.groupby(groups).transform("mean").T


def stock_signals(prices, volume, index, eligible, sector_of):
    """Wide (date x symbol) frames, one per stock signal."""
    daily = prices.pct_change(fill_method=None)
    ret = {n: prices / prices.shift(n) - 1 for n in (5, 20, 60, 120)}
    index_ret = {n: index / index.shift(n) - 1 for n in (20, 60)}
    ret20_eligible = ret[20].where(eligible)
    return {
        "ret_5": ret[5],
        "ret_20": ret[20],
        "ret_60": ret[60],
        "ret_120": ret[120],
        # Classic momentum: last 12 months, skipping the most recent month,
        # which tends to reverse rather than continue.
        "mom_12_1": prices.shift(21) / prices.shift(252) - 1,
        "rel_20": ret[20].sub(index_ret[20], axis=0),
        "rel_60": ret[60].sub(index_ret[60], axis=0),
        "sector_rel_20": ret[20] - _sector_mean(ret20_eligible, sector_of),
        "sector_ret_60": _sector_mean(ret[60].where(eligible), sector_of),
        "vol_20": daily.rolling(20).std() * np.sqrt(252),
        "vol_60": daily.rolling(60).std() * np.sqrt(252),
        "sma50_gap": prices / prices.rolling(50).mean() - 1,
        "sma200_gap": prices / prices.rolling(200).mean() - 1,
        "rsi_14": _rsi(prices),
        "drawdown_252": prices / prices.rolling(252, min_periods=60).max() - 1,
        "volume_trend": volume.rolling(20).mean() / volume.rolling(60).mean(),
        "liquidity": np.log1p((prices * volume).rolling(60, min_periods=20).median()),
    }


def market_signals(index, vix, sma50_gap, eligible):
    """One value per date describing the overall market."""
    index_daily = index.pct_change(fill_method=None)
    vix = vix.reindex(index.index).ffill()
    above_50 = (sma50_gap > 0) & eligible
    return pd.DataFrame({
        "mkt_ret_20": index / index.shift(20) - 1,
        "mkt_ret_60": index / index.shift(60) - 1,
        "mkt_sma200_gap": index / index.rolling(200).mean() - 1,
        "mkt_vol_20": index_daily.rolling(20).std() * np.sqrt(252),
        "vix": vix,
        "vix_change_20": vix / vix.shift(20) - 1,
        "breadth_50": above_50.sum(axis=1) / eligible.sum(axis=1),
    })


def with_dividends(index_return, days):
    """The index's return over `days` trading days, plus its estimated dividends."""
    return (1 + index_return) * (1 + INDEX_DIVIDEND_YIELD) ** (days / 252) - 1


def target_columns(horizon):
    """Names of the forward-looking columns for one holding period."""
    return f"fwd_return_{horizon}", f"fwd_excess_{horizon}", f"target_rank_{horizon}"


def build_panel(close, volume, vix, sector_of, horizons):
    """Long table: one row per (date, eligible stock) with features and targets.

    For each holding period h in `horizons` (trading days) there are three
    target columns: the return from buying at the close EXECUTION_LAG_DAYS
    after the signal and holding h days (fwd_return_h), the same minus the
    NIFTY 500's return with estimated dividends (fwd_excess_h), and that excess
    return's percentile rank among the day's stocks (target_rank_h).
    """
    index = close[BENCHMARK]
    prices = close.drop(columns=BENCHMARK)
    volume = volume.drop(columns=BENCHMARK, errors="ignore").reindex_like(prices)
    eligible = eligible_mask(prices, volume) & prices.notna()

    signals = stock_signals(prices, volume, index, eligible, sector_of)
    market = market_signals(index, vix, signals["sma50_gap"], eligible)

    lag = EXECUTION_LAG_DAYS
    entry, index_entry = prices.shift(-lag), index.shift(-lag)
    for h in horizons:
        fwd_return = prices.shift(-(lag + h)) / entry - 1
        index_return = with_dividends(index.shift(-(lag + h)) / index_entry - 1, h)
        return_col, excess_col, _ = target_columns(h)
        signals[return_col] = fwd_return
        signals[excess_col] = fwd_return.sub(index_return, axis=0)

    # Keep only eligible (date, stock) cells, without building the full grid.
    rows, cols = np.nonzero(eligible.to_numpy())
    columns = {name: frame.to_numpy(dtype="float64")[rows, cols] for name, frame in signals.items()}
    for name in STOCK_FEATURES:
        ranked = signals[name].where(eligible).rank(axis=1, pct=True)
        columns[f"{name}_rank"] = ranked.to_numpy(dtype="float64")[rows, cols]
    for name in MARKET_FEATURES:
        columns[name] = market[name].to_numpy(dtype="float64")[rows]

    panel = pd.DataFrame(columns, index=pd.MultiIndex.from_arrays(
        [prices.index[rows], prices.columns[cols]], names=["date", "symbol"]))
    panel = panel.replace([np.inf, -np.inf], np.nan).astype("float32")
    panel = panel.dropna(subset=["mom_12_1", "sma200_gap"])
    by_date = panel.groupby(level="date")
    for h in horizons:
        _, excess_col, rank_col = target_columns(h)
        panel[rank_col] = by_date[excess_col].rank(pct=True).astype("float32")
    panel["sector"] = panel.index.get_level_values("symbol").map(sector_of).fillna("Other")
    return panel
