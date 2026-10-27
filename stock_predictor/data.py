"""Download and cache prices and company fundamentals from Yahoo Finance."""

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from .paths import CACHE_DIR
from .universe import BENCHMARK, BENCHMARK_SYMBOL

# yfinance still uses pandas' deprecated inplace= arguments; the warnings are noise.
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

PRICE_TTL = 6 * 3600
FUNDAMENTALS_TTL = 7 * 24 * 3600
BATCH_SIZE = 100
MIN_HISTORY_DAYS = 250

FUNDAMENTAL_FIELDS = {
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
    "trailingPE": "pe",
    "priceToBook": "pb",
    "profitMargins": "profit_margin",
    "earningsGrowth": "earnings_growth",
    "revenueGrowth": "revenue_growth",
    "heldPercentInsiders": "promoter_holding",
    "marketCap": "market_cap",
}


def _is_fresh(path, ttl):
    return path.exists() and time.time() - path.stat().st_mtime < ttl


def _download(yahoo_symbols, period):
    frames = []
    for start in range(0, len(yahoo_symbols), BATCH_SIZE):
        batch = yahoo_symbols[start:start + BATCH_SIZE]
        raw = yf.download(batch, period=period, auto_adjust=True, progress=False, threads=True)
        if not raw.empty:
            frames.append(raw[["Close", "Volume"]])
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def load_prices(symbols, period="5y", refresh=False):
    """Daily close and volume for each symbol plus the NIFTY 500 benchmark.

    Returns (close, volume, missing) where missing lists symbols Yahoo had no
    usable history for (too new, suspended or renamed).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"prices_{period}.pkl"
    wanted = sorted(symbols)
    if not refresh and _is_fresh(cache, PRICE_TTL):
        cached = pd.read_pickle(cache)
        if cached["symbols"] == wanted:
            return cached["close"], cached["volume"], cached["missing"]

    yahoo = {f"{s}.NS": s for s in wanted}
    yahoo[BENCHMARK_SYMBOL] = BENCHMARK
    raw = _download(list(yahoo), period)
    if raw.empty or BENCHMARK_SYMBOL not in raw["Close"]:
        raise RuntimeError("Could not download prices from Yahoo Finance. Check your internet connection.")

    close = raw["Close"].rename(columns=yahoo)
    volume = raw["Volume"].rename(columns=yahoo)
    close = close[close[BENCHMARK].notna()].sort_index()
    close = close.loc[:, close.notna().sum() >= MIN_HISTORY_DAYS]
    volume = volume.reindex(index=close.index, columns=close.columns)
    missing = sorted(set(wanted) - set(close.columns))

    pd.to_pickle({"symbols": wanted, "close": close, "volume": volume, "missing": missing}, cache)
    return close, volume, missing


def _fetch_info(symbol):
    try:
        info = yf.Ticker(f"{symbol}.NS").info or {}
    except Exception:
        info = {}
    return {name: info.get(field) for field, name in FUNDAMENTAL_FIELDS.items()}


def load_fundamentals(symbols, refresh=False):
    """Key ratios per symbol (ROE, debt/equity, P/E, ...), cached for a week."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "fundamentals.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()

    stale = [s for s in symbols
             if refresh or s not in cache or now - cache[s]["fetched_at"] > FUNDAMENTALS_TTL]
    if stale:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol, values in zip(stale, pool.map(_fetch_info, stale)):
                got_data = any(v is not None for v in values.values())
                # A failed lookup gets fetched_at=0 so it is retried next time.
                cache[symbol] = {"fetched_at": now if got_data else 0, **values}
        path.write_text(json.dumps(cache, indent=1))

    return _to_frame(cache, symbols)


def cached_fundamentals():
    """Everything downloaded so far, used e.g. for sector P/E medians."""
    path = CACHE_DIR / "fundamentals.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    return _to_frame(cache, list(cache))


def _to_frame(cache, symbols):
    rows = {s: {k: v for k, v in cache[s].items() if k != "fetched_at"} for s in symbols}
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(FUNDAMENTAL_FIELDS.values()))
    frame = frame.apply(pd.to_numeric, errors="coerce")
    # Yahoo often lacks ROE for Indian stocks; ROE = earnings / book = (P/B) / (P/E).
    implied_roe = (frame.pb / frame.pe).where((frame.pe > 0) & (frame.pb > 0))
    frame["roe"] = frame.roe.fillna(implied_roe)
    return frame
