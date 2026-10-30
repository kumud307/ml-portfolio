import numpy as np
import pytest

from stock_predictor import backtest, time_machine
from stock_predictor.config import EXECUTION_LAG_DAYS
from stock_predictor.features import build_panel, target_columns
from stock_predictor.model import training_cutoff

HORIZON = 5
DAY = 560                     # late enough to train on, early enough to know the outcome


@pytest.fixture(scope="module")
def market(fake_market):
    close, volume, vix, sector_of = fake_market
    return close, build_panel(close, volume, vix, sector_of, horizons=(HORIZON,))


def test_travel_compares_picks_with_what_really_happened(market):
    close, panel = market
    trip = time_machine.travel(panel, close, close.index[DAY], HORIZON)

    entry, exit_ = DAY + EXECUTION_LAG_DAYS, DAY + EXECUTION_LAG_DAYS + HORIZON
    chosen = trip.picks[trip.picks.strategy == backtest.ML]
    prices = close[list(chosen.symbol)]
    expected = (prices.iloc[exit_] / prices.iloc[entry] - 1).to_numpy()
    assert np.allclose(chosen.actual_return, expected, rtol=1e-5)
    assert trip.end == close.index[exit_]


def test_travel_ignores_answers_unknown_on_the_day(market):
    close, panel = market
    as_of = close.index[DAY]
    target = target_columns(HORIZON)[2]
    dates = panel.index.get_level_values("date")
    unknown = (dates > training_cutoff(dates, as_of, HORIZON)) & panel[target].notna()
    tampered = panel.copy()
    noise = np.random.default_rng(0).random(unknown.sum()).astype("float32")
    tampered.loc[unknown, target] = noise

    original = time_machine.travel(panel, close, as_of, HORIZON).picks
    replayed = time_machine.travel(tampered, close, as_of, HORIZON).picks

    assert list(replayed.symbol) == list(original.symbol)


def test_travel_refuses_dates_it_cannot_test(market):
    close, panel = market
    with pytest.raises(ValueError, match="Too recent"):
        time_machine.travel(panel, close, close.index[-1], HORIZON)
    with pytest.raises(ValueError, match="Too early"):
        time_machine.travel(panel, close, close.index[300], HORIZON)


def test_run_many_gives_one_row_per_date_and_strategy(market):
    close, panel = market
    rows = time_machine.run_many(panel, close, HORIZON)
    assert not rows.duplicated(["date", "strategy"]).any()
    assert set(rows.strategy) == set(backtest.STRATEGIES + backtest.BENCHMARKS)
