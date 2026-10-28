"""Risk metrics, today's scores, sector comparison and the personal safety check.

Only backtested strategies decide the ranking. Company fundamentals can't be
backtested here (we only have today's figures, not past ones), so they are
shown as checks and warnings, never mixed into the score.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import strategy_scores
from .universe import BENCHMARK

FINANCIAL_SECTOR = "Financial Services"

SIGNALS = [  # (minimum percentile rank today, label)
    (0.95, "Strong candidate"),
    (0.80, "Candidate"),
    (0.50, "Watch"),
    (0.00, "Avoid for now"),
]

RISK_PROFILES = {
    # profile: (risk levels it accepts, max share of portfolio in one stock)
    "low": ({"Low"}, 0.05),
    "medium": ({"Low", "Medium"}, 0.10),
    "high": ({"Low", "Medium", "High"}, 0.15),
}


def risk_metrics(close):
    """Volatility, beta, drawdown and a typical bad month, from recent prices."""
    year = close.iloc[-252:]
    daily = year.pct_change(fill_method=None)
    index_daily = daily[BENCHMARK]
    stocks = daily.drop(columns=BENCHMARK)

    two_years = close.iloc[-504:].drop(columns=BENCHMARK)
    monthly = two_years / two_years.shift(20) - 1
    prices = year.drop(columns=BENCHMARK)

    risk = pd.DataFrame({
        "volatility": stocks.std() * np.sqrt(252),
        "beta": stocks.apply(lambda s: s.cov(index_daily)) / index_daily.var(),
        "max_drawdown": (prices / prices.cummax() - 1).min(),
        # In 1 month out of 20, the stock fell at least this much.
        "bad_month": monthly.quantile(0.05),
    })
    points = (
        (risk.volatility > 0.28).astype(int) + (risk.volatility > 0.40)
        + (risk.max_drawdown < -0.25) + (risk.max_drawdown < -0.40)
        + (risk.beta > 1.2)
    )
    risk["risk_level"] = np.select([points <= 1, points <= 3], ["Low", "Medium"], "High")
    return risk


def trend_checks(table):
    return pd.DataFrame({
        "Above 50-day average": table.sma50_gap > 0,
        "Above 200-day average": table.sma200_gap > 0,
        "Beating the market (3 months)": table.rel_60 > 0,
        "Not overbought or oversold (RSI 40-70)": table.rsi_14.between(40, 70),
    })


def sector_pe_medians(fundamentals, sector_of, min_peers=3):
    pe = fundamentals.pe.where(fundamentals.pe > 0)
    sector = pd.Series(sector_of).reindex(pe.index)
    stats = pe.groupby(sector).agg(["median", "count"])
    return stats["median"].where(stats["count"] >= min_peers)


def fundamental_checks(fundamentals, sector_of, sector_pe):
    """1.0 = pass, 0.0 = fail, NaN = no data / rule doesn't apply."""
    f = fundamentals
    sector = pd.Series(sector_of).reindex(f.index)
    sector_pe = pd.to_numeric(sector.map(sector_pe), errors="coerce")

    def rule(passed, applies):
        return passed.astype(float).where(applies)

    return pd.DataFrame({
        "Return on equity above 15%": rule(f.roe > 0.15, f.roe.notna()),
        # Debt is a bank's raw material, so the debt rule skips financials.
        "Debt below equity": rule(f.debt_to_equity < 100,
                                  f.debt_to_equity.notna() & (sector != FINANCIAL_SECTOR)),
        "P/E not above sector median": rule((f.pe > 0) & (f.pe <= sector_pe * 1.1),
                                            f.pe.notna() & sector_pe.notna()),
        "Profit margin above 10%": rule(f.profit_margin > 0.10, f.profit_margin.notna()),
        "Earnings growing": rule(f.earnings_growth > 0, f.earnings_growth.notna()),
    })


def add_fundamentals(table, fundamentals, sector_pe):
    table = table.join(fundamentals, how="left")
    checks = fundamental_checks(fundamentals, table.sector, sector_pe)
    table["fundamental_score"] = checks.mean(axis=1).reindex(table.index)
    return table


def signal_label(rank_pct):
    return next(label for floor, label in SIGNALS if rank_pct >= floor)


def score_today(latest, risk, universe, strategy):
    """Every eligible stock today, scored by every strategy and ranked by `strategy`."""
    scores = strategy_scores(latest)
    table = latest.join(risk).join(universe[["name"]])
    table["name"] = table["name"].fillna(pd.Series(table.index, index=table.index))
    for name in scores:
        table[f"score: {name}"] = scores[name]
    table["rank_pct"] = scores[strategy].rank(pct=True)
    table["signal"] = table["rank_pct"].map(signal_label)
    table["trend_score"] = trend_checks(table).mean(axis=1)
    return table.sort_values("rank_pct", ascending=False)


def sector_summary(table):
    in_top_50 = table.rank_pct >= table.rank_pct.nlargest(50).min()
    groups = table.assign(in_top_50=in_top_50).groupby("sector")
    summary = pd.DataFrame({
        "stocks": groups.size(),
        "return_1m": groups.ret_20.median(),
        "return_3m": groups.ret_60.median(),
        "vs_market_3m": groups.rel_60.median(),
        "breadth": groups.sma50_gap.agg(lambda s: (s > 0).mean()),
        "volatility": groups.volatility.median(),
        "avg_rank": groups.rank_pct.mean(),
        "top_50": groups.in_top_50.sum(),
    })
    summary["strength"] = (summary.avg_rank.rank(pct=True) * 0.4
                           + summary.vs_market_3m.rank(pct=True) * 0.3
                           + summary.breadth * 0.3)
    return summary.sort_values("strength", ascending=False)


@dataclass
class SafetyReport:
    verdict: str
    tone: str                    # "good" | "warning" | "critical"
    risk_level: str
    fits_profile: bool
    max_allocation: float        # share of total portfolio
    suggested_amount: float      # rupees
    bad_month_loss: float        # rupees on the amount above
    worst_year_loss: float       # rupees, if last year's worst drop repeats
    positives: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def safety_check(row, profile, portfolio_value, strategy, strategy_stats, news_sentiment=None):
    """Plain-language answer to: is this stock safe for *me* to buy right now?

    `strategy_stats` is the strategy's backtest summary row (hit rate, etc.).
    """
    accepted, cap = RISK_PROFILES[profile]
    fits = row.risk_level in accepted
    max_alloc = cap * (0.5 if row.risk_level == "High" else 1.0)
    amount = portfolio_value * max_alloc

    good, bad = [], []
    if row.sma200_gap > 0:
        good.append("Price is above its 200-day average (long-term uptrend)")
    else:
        bad.append("Price is below its 200-day average (long-term downtrend)")
    if row.rel_60 > 0:
        good.append(f"Beat the market by {row.rel_60:.1%} over the last 3 months")
    else:
        bad.append(f"Lagged the market by {-row.rel_60:.1%} over the last 3 months")
    if row.drawdown_252 < -0.25:
        bad.append(f"Trading {-row.drawdown_252:.0%} below its 1-year high")
    if row.rsi_14 > 75:
        bad.append(f"Looks overbought (RSI {row.rsi_14:.0f}); buying after a sharp run-up is risky")
    if row.rank_pct >= 0.80:
        good.append(f"Ranked in the top {1 - row.rank_pct:.0%} of stocks by the {strategy} strategy")
    elif row.rank_pct < 0.50:
        bad.append(f"Ranked in the bottom half of stocks by the {strategy} strategy")
    if row.beta > 1.3:
        bad.append(f"Swings {row.beta:.1f}x as much as the market (beta)")

    fundamental_notes = {
        "roe": (lambda v: v > 0.15, "Strong return on equity ({:.0%})", "Weak return on equity ({:.0%})"),
        "profit_margin": (lambda v: v > 0.10, "Healthy profit margin ({:.0%})", "Thin profit margin ({:.0%})"),
        "earnings_growth": (lambda v: v > 0, "Earnings growing ({:+.0%} YoY)", "Earnings shrinking ({:+.0%} YoY)"),
    }
    for col, (ok, good_text, bad_text) in fundamental_notes.items():
        value = row.get(col)
        if value is not None and pd.notna(value):
            (good if ok(value) else bad).append((good_text if ok(value) else bad_text).format(value))
    de = row.get("debt_to_equity")
    if de is not None and pd.notna(de) and row.sector != FINANCIAL_SECTOR and de > 150:
        bad.append(f"High debt (debt/equity {de / 100:.1f}x)")

    if news_sentiment is not None and news_sentiment < -0.2:
        bad.append("Recent news headlines are mostly negative")
    elif news_sentiment is not None and news_sentiment > 0.2:
        good.append("Recent news headlines are mostly positive")

    if not fits:
        verdict, tone = (f"Too risky for a {profile}-risk investor. This stock is rated "
                         f"{row.risk_level} risk."), "critical"
    elif row.rank_pct >= 0.80 and len(bad) <= 2:
        verdict, tone = ("Reasonable to consider, within the position limit below. "
                         "Consider buying in 2-3 parts rather than all at once."), "good"
    elif row.rank_pct >= 0.50:
        verdict, tone = "Mixed signals. Better to watch it for now than buy.", "warning"
    else:
        verdict, tone = "Avoid for now. Most signals are negative.", "critical"
    bad.append(f"Even the {strategy} strategy beat the market in only "
               f"{strategy_stats['hit_rate']:.0%} of past holding periods, so any single pick can lose")

    return SafetyReport(
        verdict=verdict,
        tone=tone,
        risk_level=row.risk_level,
        fits_profile=fits,
        max_allocation=max_alloc,
        suggested_amount=amount,
        bad_month_loss=amount * -min(row.bad_month, 0),
        worst_year_loss=amount * -min(row.max_drawdown, 0),
        positives=good,
        warnings=bad,
    )
