"""The stocks we analyse, and which of them were eligible on each date.

Candidates come from NSE's NIFTY Total Market list (~750 stocks with sectors).
On every date, only the UNIVERSE_SIZE most-traded stocks *as of that date*
are eligible. Using a fixed list of today's index members for the past would
quietly favour companies that later grew big (survivorship bias).
"""

import io
import time

import pandas as pd
import requests

from .config import LIQUIDITY_LOOKBACK, UNIVERSE_SIZE
from .paths import ASSETS_DIR, CACHE_DIR

LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
SNAPSHOT = ASSETS_DIR / "nifty_total_market_list.csv"
LIST_TTL = 7 * 24 * 3600

BENCHMARK_SYMBOL = "^CRSLDX"   # Yahoo Finance code for the NIFTY 500 index
BENCHMARK = "NIFTY500"
VIX_SYMBOL = "^INDIAVIX"       # India VIX: the market's expected volatility ("fear gauge")
VIX = "INDIAVIX"

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
    except Exception as error:
        print(f"NSE stock list download failed ({error}); using the saved copy in assets/.")
        return _parse(SNAPSHOT.read_text(encoding="utf-8"))


def eligible_mask(prices, volume, size=UNIVERSE_SIZE, lookback=LIQUIDITY_LOOKBACK):
    """True where a stock was among the `size` most-traded stocks on that date.

    Traded value = price x shares traded. Using the median over the past
    `lookback` days means each date only uses information available then.
    """
    traded_value = (prices * volume).rolling(lookback, min_periods=lookback // 2).median()
    rank = traded_value.rank(axis=1, ascending=False)
    return rank <= size
