import numpy as np
import pandas as pd

from stock_predictor.config import EXECUTION_LAG_DAYS
from stock_predictor.model import training_cutoff


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
