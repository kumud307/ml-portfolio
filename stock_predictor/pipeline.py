"""End-to-end run: stock list -> prices -> features -> model -> backtest -> today's picks."""

from dataclasses import dataclass

import pandas as pd

from . import analysis, backtest, model
from .config import ALL_HORIZONS, TOP_N
from .data import cached_fundamentals, load_fundamentals, load_prices
from .features import build_panel
from .paths import CACHE_DIR
from .universe import load_universe

# Bump this when features or the model change, so old saved results are ignored.
MODEL_VERSION = 2


@dataclass
class Market:
    universe: pd.DataFrame   # symbol -> name, sector (all candidates)
    close: pd.DataFrame      # daily closes, one column per stock + NIFTY500
    missing: list            # candidates without enough price history
    panel: pd.DataFrame      # (date, symbol) features and targets for eligible stocks
    risk: pd.DataFrame       # today's risk metrics per stock

    @property
    def as_of(self):
        return self.panel.index.get_level_values("date").max()


@dataclass
class StrategyRun:
    horizon: int
    backtest: backtest.BacktestResult
    latest: pd.DataFrame     # today's eligible stocks with the model's score


def load_market(refresh=False):
    universe = load_universe(refresh=refresh)
    close, volume, vix, missing = load_prices(list(universe.index), refresh=refresh)
    panel = build_panel(close, volume, vix, universe.sector.to_dict(), ALL_HORIZONS)
    return Market(universe, close, missing, panel, analysis.risk_metrics(close))


def run_strategy(market, horizon):
    """Backtest every strategy and score today's stocks, for one holding period.

    The slow part (walk-forward training) is saved to disk and redone only
    when new prices arrive or the model changes.
    """
    path = CACHE_DIR / f"strategy_h{horizon}.pkl"
    key = (MODEL_VERSION, str(market.as_of.date()), len(market.panel))
    if path.exists():
        saved = pd.read_pickle(path)
        if saved["key"] == key:
            return StrategyRun(horizon, saved["backtest"], saved["latest"])

    out_of_sample = model.walk_forward(market.panel, horizon)
    result = backtest.run_backtest(market.panel, out_of_sample, horizon)
    fitted = model.fit(market.panel, horizon)
    latest = model.predict_day(fitted, market.panel, market.as_of)
    pd.to_pickle({"key": key, "backtest": result, "latest": latest}, path)
    return StrategyRun(horizon, result, latest)


def today_table(market, run, strategy):
    """All eligible stocks today, ranked by `strategy`."""
    table = analysis.score_today(run.latest, market.risk, market.universe, strategy)
    table["price"] = market.close.ffill().iloc[-1].reindex(table.index)
    return table


def top_picks(market, table, strategy, n=TOP_N):
    """Today's top `n` (sector-capped) for `strategy`, with fundamentals attached."""
    symbols = backtest.select_top(table[f"score: {strategy}"], table["sector"], n)
    picks = table.loc[symbols]
    fundamentals = load_fundamentals(symbols)
    return analysis.add_fundamentals(picks, fundamentals, _sector_pe(market))


def stock_details(market, table, symbol, peers=8):
    """One stock's full row including fundamentals.

    A few sector peers are fetched too, so the P/E comparison has a baseline.
    """
    same_sector = table.index[(table.sector == table.at[symbol, "sector"]) & (table.index != symbol)]
    fundamentals = load_fundamentals([symbol, *same_sector[:peers]])
    row = analysis.add_fundamentals(table.loc[[symbol]], fundamentals.loc[[symbol]], _sector_pe(market))
    return row.iloc[0]


def _sector_pe(market):
    return analysis.sector_pe_medians(cached_fundamentals(), market.universe.sector)
