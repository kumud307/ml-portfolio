import numpy as np
import pandas as pd

from stock_predictor import analysis, tracker
from stock_predictor.universe import BENCHMARK


def test_calm_stock_is_low_risk_and_wild_stock_is_high_risk(fake_market):
    close, *_ = fake_market
    calm = close[BENCHMARK] * 1.0001          # moves exactly like the market
    wild = close["S00"] ** 3 / close["S00"].iloc[0] ** 2   # triple the swings
    risk = analysis.risk_metrics(close.assign(CALM=calm, WILD=wild))
    assert risk.at["CALM", "risk_level"] == "Low"
    assert risk.at["WILD", "risk_level"] == "High"


def _row(**overrides):
    base = dict(risk_level="High", sma200_gap=0.1, rel_60=0.05, drawdown_252=-0.1, rsi_14=55,
                rank_pct=0.97, beta=1.0, sector="IT", bad_month=-0.1, max_drawdown=-0.3)
    return pd.Series({**base, **overrides})


def test_high_risk_stock_is_not_suitable_for_careful_investor():
    report = analysis.safety_check(_row(), "low", 100_000, "Momentum", {"hit_rate": 0.6})
    assert report.tone == "critical" and not report.fits_profile


def test_high_risk_stock_gets_half_the_position_size():
    report = analysis.safety_check(_row(), "high", 100_000, "Momentum", {"hit_rate": 0.6})
    assert report.max_allocation == 0.075            # 15% cap halved
    assert report.bad_month_loss == 7_500 * 0.1


def test_tracker_records_once_and_scores_picks(fake_market, tmp_path):
    close, *_ = fake_market
    path = tmp_path / "picks.csv"
    as_of = close.index[500]
    picks = pd.DataFrame({"name": ["Stock 1", "Stock 2"], "sector": ["IT", "Banks"],
                          "price": [100.0, 200.0]}, index=["S01", "S02"])

    assert tracker.record(as_of, 5, "Momentum", picks, path)
    assert not tracker.record(as_of, 5, "Momentum", picks, path)   # no duplicates

    result = tracker.evaluate(close, path).iloc[0]
    start, end = 501, 506                                         # bought next day, held 5
    expected = (close[["S01", "S02"]].iloc[end] / close[["S01", "S02"]].iloc[start] - 1).mean()
    assert result.status == "Finished"
    assert abs(result.portfolio_return - expected) < 1e-9


def _record_two_picks(close, path):
    picks = pd.DataFrame({"name": ["Stock 1", "Stock 2"], "sector": ["IT", "Banks"],
                          "price": [100.0, 200.0]}, index=["S01", "S02"])
    tracker.record(close.index[500], 5, "Momentum", picks, path)


def test_pick_that_stops_trading_stays_in_at_its_last_price(fake_market, tmp_path):
    close, *_ = fake_market
    path = tmp_path / "picks.csv"
    _record_two_picks(close, path)
    halted = close.copy()
    halted.loc[close.index[503]:, "S02"] = np.nan

    result = tracker.evaluate(halted, path).iloc[0]

    kept = close.S01.iloc[506] / close.S01.iloc[501] - 1
    frozen = close.S02.iloc[502] / close.S02.iloc[501] - 1
    assert result.stocks == 2
    assert abs(result.portfolio_return - (kept + frozen) / 2) < 1e-9


def test_pick_with_no_price_is_flagged_not_dropped(fake_market, tmp_path):
    """Scoring the list without it would hide exactly the stocks that collapsed."""
    close, *_ = fake_market
    path = tmp_path / "picks.csv"
    _record_two_picks(close, path)

    result = tracker.evaluate(close.drop(columns="S02"), path).iloc[0]

    assert result.status == "Missing prices" and result.unpriced == 1
    assert pd.isna(result.get("portfolio_return"))


def test_summary_keeps_strategies_and_holding_periods_apart():
    finished = pd.DataFrame({"strategy": ["A", "A", "A", "B"], "horizon": [5, 5, 15, 5],
                             "vs_market": [0.10, -0.02, -0.10, 0.20]})
    summary = tracker.summarise(finished).set_index(["strategy", "horizon"])
    assert len(summary) == 3
    assert summary.at[("A", 5), "lists"] == 2
    assert summary.at[("A", 5), "won"] == 0.5
    assert abs(summary.at[("A", 5), "avg_vs_market"] - 0.04) < 1e-9
