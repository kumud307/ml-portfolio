"""Recent headlines from Indian financial news feeds, with a simple sentiment score."""

import re

import feedparser
import pandas as pd
import requests

RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.financialexpress.com/market/feed/",
    "https://www.ndtv.com/business/rss",
]

POSITIVE = {
    "surge", "surges", "rally", "rallies", "gain", "gains", "jump", "jumps", "rise", "rises",
    "soar", "soars", "record", "beat", "beats", "upgrade", "upgraded", "strong", "growth",
    "profit", "outperform", "buy", "bullish", "high", "wins", "win", "order", "orders", "expands",
}
NEGATIVE = {
    "fall", "falls", "drop", "drops", "slump", "slumps", "plunge", "plunges", "loss", "losses",
    "miss", "misses", "downgrade", "downgraded", "weak", "decline", "declines", "cut", "cuts",
    "probe", "fraud", "penalty", "sell", "bearish", "crash", "low", "slips", "tumbles", "default",
    "raid", "resigns", "lawsuit", "ban",
}
LEGAL_SUFFIX = re.compile(r"\s*(Ltd\.?|Limited|\(India\))\s*$", re.IGNORECASE)


def fetch_headlines():
    rows = []
    for url in RSS_FEEDS:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            feed = feedparser.parse(response.content)
        except Exception:
            continue
        source = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        for entry in feed.entries:
            rows.append({"source": source, "headline": entry.get("title", ""),
                         "published": entry.get("published", ""), "link": entry.get("link", "")})
    columns = ["source", "headline", "published", "link"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates("headline")


def sentiment(text):
    """-1 (all negative words) .. +1 (all positive words); 0 if neither."""
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def headlines_for(symbol, company_name, headlines):
    """Headlines that mention the company by name or by its NSE symbol."""
    name = LEGAL_SUFFIX.sub("", company_name).strip()
    by_name = headlines.headline.str.contains(rf"\b{re.escape(name)}\b", case=False, regex=True)
    # Symbols are matched case-sensitively so e.g. 'BEL' doesn't match 'bel'.
    by_symbol = headlines.headline.str.contains(rf"\b{re.escape(symbol)}\b", case=True, regex=True)
    matched = headlines[by_name | (by_symbol if len(symbol) >= 3 else False)].copy()
    matched["sentiment"] = matched.headline.map(sentiment)
    return matched
