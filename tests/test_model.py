import numpy as np
import pandas as pd
import pytest

from stock_predictor.config import EXECUTION_LAG_DAYS
from stock_predictor.features import FEATURES, build_panel, target_columns
from stock_predictor.model import fit, training_cutoff, walk_forward

HORIZON = 5


@pytest.fixture(scope="module")
def panel(fake_market):
    close, volume, vix, sector_of = fake_market
    return build_panel(close, volume, vix, sector_of, horizons=(HORIZON,))


def _scramble_answers(panel, rows):
    """Replace the training target with noise on `rows` that have one."""
    target = target_columns(HORIZON)[2]
    rows = rows & panel[target].notna()
    tampered = panel.copy()
    noise = np.random.default_rng(0).random(rows.sum()).astype("float32")
    tampered.loc[rows, target] = noise
    return tampered


def test_training_cutoff_only_allows_known_answers():
    dates = pd.bdate_range("2024-01-01", periods=100)
    as_of, horizon = dates[60], 10
    cutoff = training_cutoff(dates, as_of, horizon)
    # The last training row's answer needs prices up to cutoff + lag + horizon days...
    answer_known_on = dates[dates.get_loc(cutoff) + EXECUTION_LAG_DAYS + horizon]
    # ...which must not be after the day we're standing on.
    assert answer_known_on <= as_of
    # And it's the latest such date (we don't throw away usable data).
    assert dates[dates.get_loc(cutoff) + 1 + EXECUTION_LAG_DAYS + horizon] > as_of


def test_training_cutoff_snaps_to_trading_day():
    dates = pd.bdate_range("2024-01-01", periods=50)
    saturday = pd.Timestamp("2024-02-10")
    assert training_cutoff(dates, saturday, 5) in set(dates)
    assert np.datetime64(training_cutoff(dates, saturday, 5)) < np.datetime64(saturday)


def test_walk_forward_cannot_see_answers_from_its_future(panel):
    """Scrambling answers after a block's cutoff must not change that block's predictions."""
    target = target_columns(HORIZON)[2]
    dates = panel.index.get_level_values("date")
    labelled = np.sort(dates[panel[target].notna()].unique())
    last_block = np.array_split(labelled, 4)[-1]
    cutoff = training_cutoff(labelled, last_block[0], HORIZON)
    in_last_block = dates[dates >= last_block[0]]

    original = walk_forward(panel, HORIZON, n_splits=3)
    after_cutoff = walk_forward(_scramble_answers(panel, dates >= cutoff), HORIZON, n_splits=3)
    before_cutoff = walk_forward(_scramble_answers(panel, dates < cutoff), HORIZON, n_splits=3)

    block = original.index.get_level_values("date").isin(in_last_block)
    assert np.array_equal(original[block], after_cutoff[block])
    # The test can fail: answers the model is allowed to use do change its output.
    assert not np.array_equal(original[block], before_cutoff[block])


def test_fit_as_of_ignores_answers_not_known_yet(panel):
    dates = panel.index.get_level_values("date")
    as_of = np.sort(dates.unique())[250]
    cutoff = training_cutoff(dates, as_of, HORIZON)
    day = panel.xs(as_of, level="date")[FEATURES]

    original = fit(panel, HORIZON, as_of=as_of).predict(day)
    unknown = fit(_scramble_answers(panel, dates > cutoff), HORIZON, as_of=as_of).predict(day)
    known = fit(_scramble_answers(panel, dates <= cutoff), HORIZON, as_of=as_of).predict(day)

    assert np.array_equal(original, unknown)
    assert not np.array_equal(original, known)
