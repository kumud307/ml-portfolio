"""Turn daily prices into a (date, symbol) table of model features and targets."""

import numpy as np
import pandas as pd

from .universe import BENCHMARK

FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120",
    "rel_20", "rel_60",
    "sector_rel_20", "sector_ret_60",
    "vol_20", "vol_60",
    "sma50_gap", "sma200_gap",
    "rsi_14", "drawdown_252", "volume_trend",
]


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def _sector_mean(frame, sector_of):
    """Each cell replaced by the average of its sector on that day."""
    groups = pd.Series(sector_of).reindex(frame.columns).fillna("Other")
    return frame.T.groupby(groups).transform("mean").T


def build_panel(close, volume, sector_of, horizon=20):
    """Features for every stock and day, plus the target: did the stock beat
    the NIFTY 500 over the next `horizon` trading days?"""
    index = close[BENCHMARK]
    prices = close.drop(columns=BENCHMARK)
    volume = volume.drop(columns=BENCHMARK, errors="ignore")
    daily = prices.pct_change(fill_method=None)

    ret = {n: prices / prices.shift(n) - 1 for n in (5, 20, 60, 120)}
    index_ret = {n: index / index.shift(n) - 1 for n in (20, 60)}

    frames = {
        "ret_5": ret[5],
        "ret_20": ret[20],
        "ret_60": ret[60],
        "ret_120": ret[120],
        "rel_20": ret[20].sub(index_ret[20], axis=0),
        "rel_60": ret[60].sub(index_ret[60], axis=0),
        "sector_rel_20": ret[20] - _sector_mean(ret[20], sector_of),
        "sector_ret_60": _sector_mean(ret[60], sector_of),
        "vol_20": daily.rolling(20).std() * np.sqrt(252),
        "vol_60": daily.rolling(60).std() * np.sqrt(252),
        "sma50_gap": prices / prices.rolling(50).mean() - 1,
        "sma200_gap": prices / prices.rolling(200).mean() - 1,
        "rsi_14": _rsi(prices),
        "drawdown_252": prices / prices.rolling(252, min_periods=60).max() - 1,
        "volume_trend": volume.rolling(20).mean() / volume.rolling(60).mean(),
        "fwd_excess": (prices.shift(-horizon) / prices - 1).sub(index.shift(-horizon) / index - 1, axis=0),
    }

    panel = pd.concat({name: frame.stack() for name, frame in frames.items()}, axis=1)
    panel.index.names = ["date", "symbol"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel["target"] = (panel["fwd_excess"] > 0).astype(float).where(panel["fwd_excess"].notna())
    return panel.dropna(subset=["ret_120", "sma200_gap"])
