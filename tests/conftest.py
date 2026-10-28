"""Shared test data: a small fake market, so tests run offline in seconds."""

import numpy as np
import pandas as pd
import pytest

from stock_predictor.universe import BENCHMARK

SECTORS = ["IT", "Banks", "Pharma"]


@pytest.fixture(scope="session")
def fake_market():
    """30 stocks in 3 sectors over 600 trading days, as random walks."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=600)
    symbols = [f"S{i:02d}" for i in range(30)]
    daily = rng.normal(0.0005, 0.02, size=(len(dates), len(symbols)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=dates, columns=symbols)
    close[BENCHMARK] = close.mean(axis=1)
    volume = pd.DataFrame(rng.integers(10_000, 100_000, size=close.shape),
                          index=dates, columns=close.columns).astype(float)
    vix = pd.Series(15 + rng.normal(0, 1, len(dates)).cumsum() * 0.1, index=dates)
    sector_of = {s: SECTORS[i % 3] for i, s in enumerate(symbols)}
    return close, volume, vix, sector_of
