from itertools import product
from types import SimpleNamespace

import pandas as pd

from stock_predictor import backtest, pipeline
from stock_predictor.config import HORIZON_CHOICES


def test_every_list_is_saved_whichever_one_is_on_screen(monkeypatch):
    symbols = [f"S{i}" for i in range(12)]
    table = pd.DataFrame({f"score: {name}": range(12) for name in backtest.STRATEGIES},
                         index=symbols).assign(sector=[f"sector {i}" for i in range(12)])
    saved = []
    monkeypatch.setattr(pipeline, "run_strategy", lambda market, horizon: None)
    monkeypatch.setattr(pipeline, "today_table", lambda market, run, strategy: table)
    monkeypatch.setattr(pipeline.tracker, "record",
                        lambda as_of, horizon, strategy, picks: saved.append((horizon, strategy)))

    pipeline.save_lists(SimpleNamespace(as_of=pd.Timestamp("2026-10-28")))

    assert sorted(saved) == sorted(product(HORIZON_CHOICES, backtest.STRATEGIES))


def test_track_record_fetches_picks_that_left_the_stock_list(monkeypatch, fake_market):
    close, *_ = fake_market
    extra = close[["S01"]].rename(columns={"S01": "GONE"})
    picks = pd.DataFrame({"symbol": ["S02", "GONE"], "as_of": close.index[500]})
    monkeypatch.setattr(pipeline.tracker, "load_picks", lambda: picks)
    monkeypatch.setattr(pipeline, "load_extra_closes", lambda symbols, start: extra[symbols])
    monkeypatch.setattr(pipeline.tracker, "evaluate", lambda prices: prices)

    scored_with = pipeline.track_record(SimpleNamespace(close=close))

    assert "GONE" in scored_with.columns
