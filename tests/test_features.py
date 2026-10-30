import numpy as np
import pandas as pd
import pytest

from stock_predictor.config import EXECUTION_LAG_DAYS, INDEX_DIVIDEND_YIELD
from stock_predictor.features import FEATURES, build_panel, with_dividends
from stock_predictor.universe import BENCHMARK


def test_features_never_use_future_prices(fake_market):
    """Features on a date must be identical whether or not later prices exist.

    If cutting off the future changed any feature, the model would be
    'peeking' at prices it couldn't know on that day (look-ahead bias).
    """
    close, volume, vix, sector_of = fake_market
    full = build_panel(close, volume, vix, sector_of, horizons=(5,))
    day = close.index[450]
    cut = build_panel(close.loc[:day], volume.loc[:day], vix.loc[:day], sector_of, horizons=(5,))

    expected = full.xs(day, level="date")[FEATURES]
    actual = cut.xs(day, level="date")[FEATURES]
    pd.testing.assert_frame_equal(actual, expected)


def test_target_is_return_from_next_day(fake_market):
    close, volume, vix, sector_of = fake_market
    h = 5
    panel = build_panel(close, volume, vix, sector_of, horizons=(h,))
    day, stock = close.index[400], "S03"
    i = close.index.get_loc(day)
    entry = close[stock].iloc[i + EXECUTION_LAG_DAYS]
    exit_ = close[stock].iloc[i + EXECUTION_LAG_DAYS + h]
    assert np.isclose(panel.loc[(day, stock), f"fwd_return_{h}"], exit_ / entry - 1, rtol=1e-5)


def test_excess_return_is_measured_against_index_with_dividends(fake_market):
    close, volume, vix, sector_of = fake_market
    h = 5
    panel = build_panel(close, volume, vix, sector_of, horizons=(h,))
    day, stock = close.index[400], "S03"
    i = close.index.get_loc(day) + EXECUTION_LAG_DAYS
    index = close[BENCHMARK]
    index_return = with_dividends(index.iloc[i + h] / index.iloc[i] - 1, h)
    stock_return = panel.loc[(day, stock), f"fwd_return_{h}"]
    assert np.isclose(panel.loc[(day, stock), f"fwd_excess_{h}"], stock_return - index_return,
                      atol=1e-5)
    assert with_dividends(0.0, 252) == pytest.approx(INDEX_DIVIDEND_YIELD)


def test_last_days_have_no_target(fake_market):
    """The final days can't have a known future return yet."""
    close, volume, vix, sector_of = fake_market
    panel = build_panel(close, volume, vix, sector_of, horizons=(5,))
    last_day = panel.xs(close.index[-1], level="date")
    assert last_day["fwd_return_5"].isna().all()
    assert last_day[FEATURES].notna().any().all()


def test_ranks_are_between_0_and_1(fake_market):
    close, volume, vix, sector_of = fake_market
    panel = build_panel(close, volume, vix, sector_of, horizons=(5,))
    ranks = panel[[c for c in FEATURES if c.endswith("_rank")]]
    assert ranks.min().min() > 0 and ranks.max().max() <= 1
