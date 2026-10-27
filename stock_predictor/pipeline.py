"""End-to-end run: stock list -> prices -> features -> model -> scores."""

from dataclasses import dataclass

import pandas as pd

from . import analysis, model
from .data import cached_fundamentals, load_fundamentals, load_prices
from .features import build_panel
from .paths import CACHE_DIR
from .universe import load_universe


@dataclass
class MarketAnalysis:
    universe: pd.DataFrame   # symbol -> name, sector
    close: pd.DataFrame      # daily closes, one column per stock + benchmark
    missing: list            # symbols without enough price history
    table: pd.DataFrame      # latest features, ML probability, risk, scores
    backtest: model.Backtest
    as_of: pd.Timestamp
    horizon: int


def _trained_model(panel, close, horizon):
    """Fit (and backtest) once per new trading day; reuse the saved model otherwise."""
    path = CACHE_DIR / f"model_h{horizon}.pkl"
    key = (str(close.index[-1].date()), tuple(close.columns))
    if path.exists():
        saved = pd.read_pickle(path)
        if saved["key"] == key:
            return saved["model"], saved["backtest"]
    backtest = model.walk_forward(panel, horizon)
    fitted = model.fit(panel)
    pd.to_pickle({"key": key, "model": fitted, "backtest": backtest}, path)
    return fitted, backtest


def run(horizon=20, refresh=False):
    universe = load_universe(refresh=refresh)
    close, volume, missing = load_prices(list(universe.index), refresh=refresh)
    panel = build_panel(close, volume, universe.sector.to_dict(), horizon)
    fitted, backtest = _trained_model(panel, close, horizon)

    latest = model.predict_latest(fitted, panel)
    table = analysis.score_stocks(latest, analysis.risk_metrics(close), universe)
    table["price"] = close.ffill().iloc[-1].reindex(table.index)
    return MarketAnalysis(universe, close, missing, table, backtest, close.index[-1], horizon)


def _sector_pe(result):
    return analysis.sector_pe_medians(cached_fundamentals(), result.universe.sector)


def top_picks(result, shortlist_size=50):
    """Shortlist by ML + trend, then rank the shortlist with fundamentals added."""
    shortlist = result.table.head(shortlist_size)
    fundamentals = load_fundamentals(list(shortlist.index))
    return analysis.add_fundamentals(shortlist, fundamentals, _sector_pe(result))


def stock_details(result, symbol, peers=8):
    """One stock's full row, including fundamentals and final score.

    A few sector peers are fetched too, so the P/E comparison has a baseline.
    """
    table = result.table
    peer_symbols = table.index[(table.sector == table.at[symbol, "sector"]) & (table.index != symbol)]
    fundamentals = load_fundamentals([symbol, *peer_symbols[:peers]])
    row = analysis.add_fundamentals(table.loc[[symbol]], fundamentals.loc[[symbol]], _sector_pe(result))
    return row.iloc[0]
