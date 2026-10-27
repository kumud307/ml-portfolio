"""The stocks we analyse: NSE's NIFTY 500 list, with each company's sector.

The NIFTY 500 covers about 95% of the NSE's traded value, so it represents the
broad Indian market across large, mid and small caps.
"""

import io
import time

import pandas as pd
import requests

from .paths import ASSETS_DIR, CACHE_DIR

LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
SNAPSHOT = ASSETS_DIR / "nifty500_list.csv"
LIST_TTL = 7 * 24 * 3600

BENCHMARK_SYMBOL = "^CRSLDX"   # Yahoo Finance code for the NIFTY 500 index
BENCHMARK = "NIFTY500"

SHORT_SECTOR_NAMES = {
    "Fast Moving Consumer Goods": "FMCG",
    "Automobile and Auto Components": "Auto",
    "Oil Gas & Consumable Fuels": "Oil & Gas",
    "Media Entertainment & Publication": "Media",
    "Information Technology": "IT",
}


def _parse(csv_text):
    raw = pd.read_csv(io.StringIO(csv_text))
    return pd.DataFrame({
        "symbol": raw["Symbol"].str.strip(),
        "name": raw["Company Name"].str.strip(),
        "sector": raw["Industry"].str.strip().replace(SHORT_SECTOR_NAMES),
    }).set_index("symbol")


def load_universe(refresh=False):
    """Symbol-indexed table of company name and sector.

    Downloads the current list from NSE weekly; falls back to the copy saved
    in assets/ when NSE can't be reached.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "universe.csv"
    if not refresh and cache.exists() and time.time() - cache.stat().st_mtime < LIST_TTL:
        return _parse(cache.read_text(encoding="utf-8"))
    try:
        response = requests.get(LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        response.raise_for_status()
        universe = _parse(response.text)
        cache.write_text(response.text, encoding="utf-8")
        return universe
    except Exception:
        return _parse(SNAPSHOT.read_text(encoding="utf-8"))
