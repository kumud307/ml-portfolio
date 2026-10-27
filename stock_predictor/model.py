"""Gradient-boosted model: probability a stock beats the NIFTY 500 over the horizon."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .features import FEATURES

# Targets on neighbouring days overlap almost entirely, so training on every
# 5th day loses little information and makes fitting ~5x faster.
TRAIN_EVERY_N_DAYS = 5


def _new_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=0,
    )


def _thin(data):
    dates = data.index.get_level_values("date")
    keep = dates.isin(np.sort(dates.unique())[::TRAIN_EVERY_N_DAYS])
    return data[keep]


@dataclass
class Backtest:
    auc: float              # 0.5 = coin flip, 1.0 = perfect
    accuracy: float
    base_rate: float        # share of cases where the stock actually beat the index
    top_pick_excess: float  # avg excess return of the model's top 20% picks
    avg_excess: float       # avg excess return of every stock
    test_from: pd.Timestamp
    test_to: pd.Timestamp

    @property
    def has_edge(self):
        return self.auc >= 0.53 and self.top_pick_excess > self.avg_excess


def walk_forward(panel, horizon, n_splits=5):
    """Train on the past, test on the following unseen period, repeat.

    A gap of `horizon` days between training and testing stops the targets of
    the last training rows from overlapping the test period (look-ahead leak).
    """
    data = panel.dropna(subset=["target"])
    dates = data.index.get_level_values("date")
    unique = np.sort(dates.unique())
    chunks = np.array_split(unique, n_splits + 1)

    tested = []
    for chunk in chunks[1:]:
        cutoff = unique[max(np.searchsorted(unique, chunk[0]) - horizon, 0)]
        train = _thin(data[dates < cutoff])
        test = data[(dates >= chunk[0]) & (dates <= chunk[-1])]
        if len(train) < 1000 or test.empty:
            continue
        model = _new_model().fit(train[FEATURES], train["target"])
        tested.append(test.assign(prob=model.predict_proba(test[FEATURES])[:, 1]))

    result = pd.concat(tested)
    pct = result.groupby(level="date")["prob"].rank(pct=True)
    test_dates = result.index.get_level_values("date")
    return Backtest(
        auc=roc_auc_score(result["target"], result["prob"]),
        accuracy=accuracy_score(result["target"], result["prob"] > 0.5),
        base_rate=result["target"].mean(),
        top_pick_excess=result.loc[pct > 0.8, "fwd_excess"].mean(),
        avg_excess=result["fwd_excess"].mean(),
        test_from=test_dates.min(),
        test_to=test_dates.max(),
    )


def fit(panel):
    labelled = _thin(panel.dropna(subset=["target"]))
    return _new_model().fit(labelled[FEATURES], labelled["target"])


def predict_latest(model, panel):
    """Score each stock on the most recent trading day."""
    last_day = panel.index.get_level_values("date").max()
    latest = panel.xs(last_day, level="date").copy()
    latest["ml_prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    return latest
