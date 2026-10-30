import numpy as np
import pandas as pd
import pytest

from stock_predictor import backtest
from stock_predictor.config import COST_ROUND_TRIP
from stock_predictor.features import build_panel


def test_select_top_respects_sector_cap():
    scores = pd.Series({"A": 9, "B": 8, "C": 7, "D": 6, "E": 5})
    sectors = pd.Series({"A": "IT", "B": "IT", "C": "IT", "D": "Banks", "E": "Banks"})
    assert backtest.select_top(scores, sectors, n=3, max_per_sector=2) == ["A", "B", "D"]


def test_select_top_keeps_held_stock_inside_buffer():
    scores = pd.Series({s: 10 - i for i, s in enumerate("ABCDEF")})
    sectors = pd.Series({s: s for s in "ABCDEF"})   # every stock its own sector
    # E ranks 5th: outside the top 2, but inside the buffer of 5, so it's kept.
    assert backtest.select_top(scores, sectors, n=2, held={"E"}, buffer=5) == ["E", "A"]
    # With a buffer of 3, E has fallen too far and is replaced.
    assert backtest.select_top(scores, sectors, n=2, held={"E"}, buffer=3) == ["A", "B"]


@pytest.fixture(scope="module")
def panel(fake_market):
    close, volume, vix, sector_of = fake_market
    return build_panel(close, volume, vix, sector_of, horizons=(5,))


def test_perfect_foresight_wins(panel):
    """Sanity check of the engine: a 'model' that knows the future must win big."""
    cheat = panel["fwd_excess_5"]
    result = backtest.run_backtest(panel, cheat, horizon=5, top_n=5, buffer=5)
    assert result.summary.loc[backtest.ML, "ic"] > 0.95
    assert result.summary.loc[backtest.ML, "cagr"] > result.summary.loc[backtest.AVERAGE, "cagr"]
    assert result.summary.loc[backtest.ML, "hit_rate"] > 0.9


def test_random_scores_have_no_skill(panel):
    noise = pd.Series(np.random.default_rng(0).random(len(panel)), index=panel.index)
    result = backtest.run_backtest(panel, noise, horizon=5, top_n=5)
    assert abs(result.summary.loc[backtest.ML, "ic"]) < 0.05


def test_only_strategies_have_an_after_tax_figure(panel):
    noise = pd.Series(np.random.default_rng(2).random(len(panel)), index=panel.index)
    summary = backtest.run_backtest(panel, noise, horizon=5, top_n=5).summary
    assert summary.loc[backtest.STRATEGIES, "cagr_after_tax"].notna().all()
    assert summary.loc[backtest.BENCHMARKS, "cagr_after_tax"].isna().all()


def test_buffer_covering_whole_market_never_trades(panel):
    """Documents the pitfall: a buffer as big as the market freezes the portfolio."""
    cheat = panel["fwd_excess_5"]
    periods = backtest.run_backtest(panel, cheat, horizon=5, top_n=5, buffer=1000).periods
    assert periods[periods.strategy == backtest.ML].turnover.iloc[1:].eq(0).all()


def test_costs_are_charged_on_turnover(panel):
    noise = pd.Series(np.random.default_rng(1).random(len(panel)), index=panel.index)
    periods = backtest.run_backtest(panel, noise, horizon=5, top_n=5).periods
    expected = periods.gross - COST_ROUND_TRIP * periods.turnover
    assert np.allclose(periods.net, expected)
    first = periods[periods.strategy == backtest.ML].iloc[0]
    assert first.turnover == 1.0     # the first period buys everything
