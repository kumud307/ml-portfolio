"""Gradient-boosted ranking model.

For every eligible stock on a date, the model predicts where its return over
the next `horizon` days will rank among all eligible stocks (0 = worst,
1 = best). Sorting by that prediction gives the top-N picks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import EXECUTION_LAG_DAYS
from .features import FEATURES, target_columns

# Targets on neighbouring days overlap almost entirely, so training on every
# 5th day loses little information and makes fitting ~5x faster.
TRAIN_EVERY_N_DAYS = 5


def _new_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=500,      # big leaves = only learn patterns seen many times
        l2_regularization=1.0,
        random_state=0,
    )


def _every_nth_date(data, n=TRAIN_EVERY_N_DAYS):
    dates = data.index.get_level_values("date")
    return data[dates.isin(np.sort(dates.unique())[::n])]


def training_cutoff(dates, as_of, horizon):
    """Last date whose target was fully known on `as_of`.

    A row's target needs prices up to EXECUTION_LAG_DAYS + horizon trading days
    after its date. Training on anything later would peek into the future.
    """
    dates = np.sort(pd.Index(dates).unique())
    position = np.searchsorted(dates, np.datetime64(as_of), side="right") - 1
    return dates[max(position - (horizon + EXECUTION_LAG_DAYS), 0)]


def fit(panel, horizon, as_of=None):
    """Train on every row whose answer was known by `as_of` (default: today)."""
    target = target_columns(horizon)[2]
    labelled = panel.dropna(subset=[target])
    if as_of is not None:
        dates = panel.index.get_level_values("date")
        cutoff = training_cutoff(dates, as_of, horizon)
        labelled = labelled[labelled.index.get_level_values("date") <= cutoff]
    labelled = _every_nth_date(labelled)
    return _new_model().fit(labelled[FEATURES], labelled[target])


def walk_forward(panel, horizon, n_splits=12):
    """Out-of-sample predictions for most of the history.

    The dates are cut into n_splits + 1 blocks. For each block after the
    first, a model is trained only on data whose answers were known before the
    block starts, then predicts that block. So every prediction is made by a
    model that never saw its answer.
    """
    target = target_columns(horizon)[2]
    data = panel.dropna(subset=[target])
    dates = data.index.get_level_values("date")
    unique = np.sort(dates.unique())

    predictions = []
    for block in np.array_split(unique, n_splits + 1)[1:]:
        cutoff = training_cutoff(unique, block[0], horizon)
        train = _every_nth_date(data[dates < cutoff])
        test = data[(dates >= block[0]) & (dates <= block[-1])]
        model = _new_model().fit(train[FEATURES], train[target])
        predictions.append(pd.Series(model.predict(test[FEATURES]), index=test.index))
    return pd.concat(predictions).rename("ml_score")


def predict_day(model, panel, date):
    """Score every eligible stock on one trading day."""
    day = panel.xs(date, level="date").copy()
    day["ml_score"] = model.predict(day[FEATURES])
    return day
