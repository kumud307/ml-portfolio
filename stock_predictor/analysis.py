"""Risk metrics, scoring rules, sector comparison and the personal safety check."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .universe import BENCHMARK

FINANCIAL_SECTOR = "Financial Services"

# The ML signal and the trend pick a shortlist from all stocks; company
# fundamentals (slow to download) are then fetched only for that shortlist.
SHORTLIST_WEIGHTS = {"ml_rank": 0.6, "trend_score": 0.4}
FINAL_WEIGHTS = {"ml_rank": 0.45, "trend_score": 0.25, "fundamental_score": 0.30}

SIGNALS = [  # (minimum score, label)
    (0.85, "Strong candidate"),
    (0.70, "Candidate"),
    (0.45, "Watch"),
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


def trend_checks(latest):
    return pd.DataFrame({
        "Above 50-day average": latest.sma50_gap > 0,
        "Above 200-day average": latest.sma200_gap > 0,
        "Beating the market (3 months)": latest.rel_60 > 0,
        "Not overbought or oversold (RSI 40-70)": latest.rsi_14.between(40, 70),
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


def _weighted(frame, weights):
    """Weighted average that ignores missing components."""
    parts = frame[list(weights)]
    w = pd.DataFrame({k: parts[k].notna() * v for k, v in weights.items()})
    return (parts.fillna(0) * w).sum(axis=1) / w.sum(axis=1)


def signal_label(score):
    return next(label for floor, label in SIGNALS if score >= floor)


def score_stocks(latest, risk, universe):
    """All stocks with ML probability, trend score, risk and shortlist score."""
    table = latest.join(risk).join(universe[["name", "sector"]])
    table["ml_rank"] = table.ml_prob.rank(pct=True)
    table["trend_score"] = trend_checks(table).mean(axis=1)
    table["shortlist_score"] = _weighted(table, SHORTLIST_WEIGHTS)
    return table.sort_values("shortlist_score", ascending=False)


def add_fundamentals(table, fundamentals, sector_pe):
    """Final score for the rows that have fundamentals downloaded."""
    table = table.join(fundamentals, how="left")
    checks = fundamental_checks(fundamentals, table.sector, sector_pe)
    table["fundamental_score"] = checks.mean(axis=1).reindex(table.index)
    table["score"] = _weighted(table, FINAL_WEIGHTS)
    table["signal"] = table.score.map(signal_label)
    return table.sort_values("score", ascending=False)


def sector_summary(table):
    groups = table.groupby("sector")
    summary = pd.DataFrame({
        "stocks": groups.size(),
        "return_1m": groups.ret_20.median(),
        "return_3m": groups.ret_60.median(),
        "vs_market_3m": groups.rel_60.median(),
        "breadth": groups.sma50_gap.agg(lambda s: (s > 0).mean()),
        "volatility": groups.volatility.median(),
        "ml_outperform_prob": groups.ml_prob.mean(),
        "trend_score": groups.trend_score.mean(),
    })
    summary["strength"] = _weighted(
        summary.assign(ml_rank=summary.ml_outperform_prob.rank(pct=True),
                       momentum_rank=summary.vs_market_3m.rank(pct=True)),
        {"ml_rank": 0.4, "momentum_rank": 0.3, "breadth": 0.3},
    )
    return summary.sort_values("strength", ascending=False)


@dataclass
class SafetyReport:
    verdict: str
    tone: str                    # "good" | "warning" | "critical"
    risk_level: str
    fits_profile: bool
    max_allocation: float        # share of total portfolio
    suggested_amount: float      # rupees, capped by budget
    bad_month_loss: float        # rupees on the amount above
    worst_year_loss: float       # rupees, if last year's worst drop repeats
    positives: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def safety_check(row, profile, portfolio_value, backtest, news_sentiment=None):
    """Plain-language answer to: is this stock safe for *me* to buy right now?"""
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
    if row.ml_prob >= 0.55:
        good.append(f"Model gives a {row.ml_prob:.0%} chance of beating the market next month")
    elif row.ml_prob < 0.45:
        bad.append(f"Model gives only a {row.ml_prob:.0%} chance of beating the market next month")
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

    score = row.get("score")
    if score is None or pd.isna(score):
        score = row.shortlist_score
    if not fits:
        verdict, tone = (f"Too risky for a {profile}-risk investor. This stock is rated "
                         f"{row.risk_level} risk."), "critical"
    elif score >= 0.70 and len(bad) <= 2:
        verdict, tone = ("Reasonable to consider, within the position limit below. "
                         "Consider buying in 2-3 parts rather than all at once."), "good"
    elif score >= 0.45:
        verdict, tone = "Mixed signals. Better to watch it for now than buy.", "warning"
    else:
        verdict, tone = "Avoid for now. Most signals are negative.", "critical"
    if not backtest.has_edge:
        bad.append("The model's past accuracy is close to a coin flip, so weight its signal lightly")

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
