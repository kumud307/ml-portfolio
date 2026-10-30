"""NSE Stock Analyser - Streamlit dashboard.

Run with:  streamlit run app.py
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_predictor import analysis, news, pipeline, time_machine, tracker
from stock_predictor.backtest import (AVERAGE, BENCHMARKS, BLEND, INDEX, ML, MOMENTUM, MOMENTUM_3M,
                                      REVERSAL, STRATEGIES, STRATEGY_NOTES)
from stock_predictor.config import (COST_ROUND_TRIP, HORIZON_CHOICES, HORIZON_DAYS, MAX_PER_SECTOR,
                                    SHORT_TERM_TAX, TIME_MACHINE_HORIZONS, TOP_N, UNIVERSE_SIZE)
from stock_predictor.universe import BENCHMARK

st.set_page_config(page_title="NSE Stock Analyser", page_icon="📈", layout="wide")

# One fixed colour per strategy, so a strategy looks the same in every chart
# (colour-blind-checked reference palette; the index is a neutral grey).
COLOURS = {ML: "#2a78d6", MOMENTUM: "#eb6834", BLEND: "#1baf7a", MOMENTUM_3M: "#eda100",
           REVERSAL: "#e87ba4", AVERAGE: "#6250d6", INDEX: "#898781"}
BLUE, RED = "#2a78d6", "#e34948"

PROFILE_LABELS = {
    "low": "Low: I can't afford big losses",
    "medium": "Medium: some ups and downs are fine",
    "high": "High: I can handle large swings",
}
DISCLAIMER = (
    "This is an educational project, not financial advice. Its predictions are rough "
    "probabilities from past price patterns, and past performance doesn't guarantee future "
    "results. Do your own research or talk to a SEBI-registered adviser before investing."
)


def pct(value, signed=True, digits=1):
    if pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def rupees(value):
    return f"₹{value:,.0f}"


# ---------------------------------------------------------------- data loading

@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def load_market(refresh_count):
    return pipeline.load_market(refresh=refresh_count > 0)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def load_strategy(horizon, refresh_count):
    return pipeline.run_strategy(load_market(refresh_count), horizon)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_today(horizon, strategy, refresh_count):
    market = load_market(refresh_count)
    return pipeline.today_table(market, load_strategy(horizon, refresh_count), strategy)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_picks(horizon, strategy, refresh_count):
    market = load_market(refresh_count)
    return pipeline.top_picks(market, load_today(horizon, strategy, refresh_count), strategy)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_stock(symbol, horizon, strategy, refresh_count):
    market = load_market(refresh_count)
    return pipeline.stock_details(market, load_today(horizon, strategy, refresh_count), symbol)


@st.cache_data(show_spinner=False)
def travel(date, horizon, refresh_count):
    market = load_market(refresh_count)
    return time_machine.travel(market.panel, market.close, date, horizon)


@st.cache_data(show_spinner=False)
def travel_many(horizon, refresh_count):
    market = load_market(refresh_count)
    return time_machine.run_many(market.panel, market.close, horizon)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def save_lists(refresh_count):
    pipeline.save_lists(load_market(refresh_count))


@st.cache_data(ttl=1800, show_spinner=False)
def load_headlines():
    return news.fetch_headlines()


# ---------------------------------------------------------------- sidebar

st.session_state.setdefault("refresh_count", 0)
refresh_count = st.session_state.refresh_count

with st.sidebar:
    st.header("📈 NSE Stock Analyser")
    st.caption(f"Ranks the {UNIVERSE_SIZE} most-traded NSE stocks and tests every idea "
               "on past data before trusting it.")
    horizon = st.radio("Holding period", list(HORIZON_CHOICES), format_func=HORIZON_CHOICES.get,
                       index=list(HORIZON_CHOICES).index(HORIZON_DAYS))

with st.spinner("Loading prices and building features (first run takes 2-3 minutes)..."):
    try:
        market = load_market(refresh_count)
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
with st.spinner(f"Testing every strategy on {HORIZON_CHOICES[horizon]} holding periods "
                "(about a minute the first time each day)..."):
    run = load_strategy(horizon, refresh_count)
with st.spinner("Saving today's lists for every strategy (about a minute the first time each day)..."):
    save_lists(refresh_count)
bt = run.backtest
summary = bt.summary

with st.sidebar:
    strategy = st.selectbox(
        "Stock-picking strategy", STRATEGIES, index=STRATEGIES.index(bt.best_strategy),
        format_func=lambda s: f"{s} ⭐" if s == bt.best_strategy else s,
        help="⭐ = best risk-adjusted return (Sharpe ratio) in the backtest. "
             "Choosing the past winner can itself be luck, so check the Backtest tab.")
    st.caption(STRATEGY_NOTES[strategy])
    if st.button("🔄 Refresh data now", width="stretch"):
        st.session_state.refresh_count += 1
        st.cache_data.clear()
        st.rerun()
    st.caption("Prices refresh automatically every 6 hours; company figures weekly.")
    st.divider()
    st.warning(DISCLAIMER, icon="⚠️")

today = load_today(horizon, strategy, refresh_count)
stats = summary.loc[strategy]
as_of = market.as_of
benchmark = market.close[BENCHMARK]

# ---------------------------------------------------------------- header

st.title("NSE Stock Analyser")
st.caption(f"Prices as of **{as_of:%d %b %Y}** · {len(today)} eligible stocks · "
           f"holding period **{HORIZON_CHOICES[horizon]}** · strategy **{strategy}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("NIFTY 500: last month", pct(benchmark.iloc[-1] / benchmark.iloc[-21] - 1))
c2.metric("NIFTY 500: last 3 months", pct(benchmark.iloc[-1] / benchmark.iloc[-61] - 1))
c3.metric("Stocks above their 50-day average", pct((today.sma50_gap > 0).mean(), signed=False),
          help="Market breadth. Above 50% means most stocks are in short-term uptrends.")
c4.metric(f"{strategy}: yearly return in backtest", pct(stats.cagr, signed=False),
          delta=f"{stats.cagr - summary.loc[INDEX, 'cagr']:+.1%} vs NIFTY 500",
          help="After trading costs, before tax. See the Backtest tab.")

tabs = st.tabs(["🏆 Top 10 picks", "📊 Backtest", "⏳ Time machine", "🏭 Market segments",
                "🛡️ Is it safe for me?", "📒 Live track record"])

# ---------------------------------------------------------------- top 10

with tabs[0]:
    st.subheader(f"Top {TOP_N} stocks for the next {HORIZON_CHOICES[horizon]}")
    st.caption(f"Strategy: **{strategy}** ({STRATEGY_NOTES[strategy].lower()}). At most "
               f"{MAX_PER_SECTOR} stocks per sector. Hold for {horizon} trading days, then check "
               "the new list.")

    st.info(
        f"**Backtest of this strategy** ({bt.start:%b %Y} to {bt.end:%b %Y}, "
        f"{len(bt.equity)} periods of {horizon} days, tested only on data the model never saw): "
        f"**{pct(stats.cagr, False)} a year** after costs ({pct(stats.cagr_after_tax, False)} after "
        f"tax) vs **{pct(summary.loc[INDEX, 'cagr'], False)}** for the NIFTY 500. It beat the "
        f"market in **{stats.hit_rate:.0%}** of periods; its worst single period was "
        f"**{pct(stats.worst_period)}**, and its worst fall from a peak was "
        f"**{pct(stats.max_drawdown)}**.", icon="📊")

    with st.spinner("Fetching company figures for the picks..."):
        picks = load_picks(horizon, strategy, refresh_count)
    budget = st.number_input("Money you want to invest in this list (₹)", min_value=10000,
                             value=100000, step=10000)
    per_stock = budget / max(len(picks), 1)
    picks = picks.assign(amount=per_stock, shares=(per_stock // picks.price).astype(int))

    st.dataframe(
        picks[["name", "sector", "price", "shares", "signal", "risk_level", "rel_60",
               "volatility", "fundamental_score", "pe", "roe"]],
        width="stretch",
        column_config={
            "name": "Company",
            "sector": "Sector",
            "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
            "shares": st.column_config.NumberColumn(
                "Shares to buy", help=f"Equal split: {rupees(per_stock)} per stock"),
            "signal": "Signal",
            "risk_level": "Risk",
            "rel_60": st.column_config.NumberColumn("vs market (3m)", format="percent"),
            "volatility": st.column_config.NumberColumn("Volatility", format="percent"),
            "fundamental_score": st.column_config.ProgressColumn(
                "Fundamentals", min_value=0, max_value=1, format="%.2f",
                help="Share of quality checks passed (ROE, debt, P/E vs sector, margin, "
                     "earnings growth). Shown for information; not used in the ranking."),
            "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
            "roe": st.column_config.NumberColumn("ROE", format="percent"),
        },
    )
    high_risk = int((picks.risk_level == "High").sum())
    weak = int((picks.fundamental_score < 0.5).sum())
    if high_risk:
        st.warning(f"{high_risk} of these stocks are rated **High** risk. For a careful "
                   "investor, consider skipping them or investing less in them.", icon="⚠️")
    if weak:
        st.warning(f"{weak} picks pass fewer than half of the company-quality checks. "
                   "Look them up in **Is it safe for me?** before buying.", icon="🔍")

    st.caption("Every strategy's list is saved each day. See how earlier lists did in the "
               "**Live track record** tab.")
    st.download_button("Download this list (CSV)", picks.to_csv().encode("utf-8"),
                       file_name=f"top{TOP_N}_{horizon}d_{as_of:%Y-%m-%d}.csv", mime="text/csv")

# ---------------------------------------------------------------- backtest

with tabs[1]:
    st.subheader("How would each strategy have done?")
    st.caption(
        f"Every {horizon} trading days from {bt.start:%b %Y} to {bt.end:%b %Y}: buy the top "
        f"{TOP_N} stocks (max {MAX_PER_SECTOR} per sector) the day after the signal, hold "
        f"{horizon} days, repeat. A stock already held is kept while it stays in the top 30. "
        f"Trading costs of {COST_ROUND_TRIP:.1%} per round trip are deducted; 'after tax' "
        f"applies {SHORT_TERM_TAX:.0%} short-term capital-gains tax to each year's profit. The ML "
        "model is retrained over time and only ever predicts periods it never saw.")

    table = summary.copy()
    table.index.name = "Strategy"
    st.dataframe(
        table.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={
            "avg_period_return": None, "volatility": None, "ic_t_stat": None,
            "cagr": st.column_config.NumberColumn("Yearly return", format="percent",
                                                  help="Compound annual growth, after costs"),
            "cagr_after_tax": st.column_config.NumberColumn(
                "After tax", format="percent",
                help="Blank for the benchmarks: buy-and-hold is taxed only when finally sold."),
            "avg_excess": st.column_config.NumberColumn(
                "Avg vs NIFTY 500 per period", format="percent"),
            "hit_rate": st.column_config.NumberColumn(
                "Periods beating NIFTY 500", format="percent"),
            "sharpe": st.column_config.NumberColumn(
                "Sharpe ratio", format="%.2f",
                help="Return above a 6.5% bank rate, per unit of risk. Above 1 is good."),
            "max_drawdown": st.column_config.NumberColumn(
                "Worst fall from peak", format="percent"),
            "worst_period": st.column_config.NumberColumn("Worst period", format="percent"),
            "turnover": st.column_config.NumberColumn(
                "Turnover", format="percent", help="Share of the portfolio replaced each period"),
            "ic": st.column_config.NumberColumn(
                "IC", format="%.3f",
                help="Information coefficient: rank correlation between the score and the "
                     "actual return. 0 = no skill; 0.03-0.05 is typical of real signals."),
        },
    )

    shown = st.multiselect("Strategies on the chart", STRATEGIES + BENCHMARKS,
                           default=list(dict.fromkeys([strategy, ML, AVERAGE, INDEX])))
    fig = go.Figure()
    for name in [s for s in STRATEGIES + BENCHMARKS if s in shown]:
        curve = bt.equity[name] * 100000
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve, name=name, mode="lines",
            line=dict(color=COLOURS[name], width=2, dash="dot" if name == INDEX else "solid"),
            hovertemplate=f"{name}: ₹%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title="Growth of ₹1 lakh, after trading costs (log scale)", height=420,
                      hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(type="log", tickprefix="₹"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "**Read these results with care.**\n"
        "- **Survivorship bias is large here.** Every stock tested is on today's NSE list, so "
        "companies that collapsed, were delisted or shrank off it are missing from the past. That "
        "flatters every strategy, momentum most. Results fall sharply on bigger, more traded "
        "stocks (see the research notes).\n"
        "- **The NIFTY 500 figures include an estimated 1% a year of dividends**, because the "
        "index level leaves them out while stock prices include them.\n"
        "- **The start day matters.** Beginning the same cycle a few days later can change a "
        "strategy's yearly return by several points, and the ML model's by far more. Treat these "
        "as rough figures (see the research notes).\n"
        "- **Choosing the best of 5 strategies** after seeing their results is itself a bit of "
        "luck. Expect future returns to be lower.\n"
        "- 2020-2024 was an unusually strong rally for Indian small and mid caps. Strategies "
        "that rode it (momentum) did very well, and in early 2025 they fell hard (try 31 Dec 2024 "
        "in the Time machine).", icon="⚠️")

# ---------------------------------------------------------------- time machine

with tabs[2]:
    st.subheader("⏳ Time machine: go back in time and check the picks")
    st.caption("Pick a past date. The model is trained **only on what was known that day**, picks "
               "its top 10, and then we fast-forward to see what actually happened.")

    t1, t2 = st.columns([1, 2])
    tm_horizon = t2.radio("Hold for", list(TIME_MACHINE_HORIZONS), horizontal=True,
                          format_func=TIME_MACHINE_HORIZONS.get, index=2)
    possible = time_machine.available_dates(market.panel, tm_horizon)
    tm_date = t1.date_input("Travel back to", value=min(dt.date(2024, 12, 31), possible[-1].date()),
                            min_value=possible[0].date(), max_value=possible[-1].date())

    with st.spinner("Training the model on data up to that date..."):
        trip = travel(pd.Timestamp(tm_date), tm_horizon, refresh_count)

    st.markdown(f"Picked on **{trip.as_of:%d %b %Y}**, bought the next day, held for "
                f"{TIME_MACHINE_HORIZONS[tm_horizon]} until **{trip.end:%d %b %Y}**. "
                f"{trip.stocks} stocks were eligible.")

    results = trip.summary
    order = results["avg_return"].sort_values().index
    fig = go.Figure(go.Bar(
        x=results.loc[order, "avg_return"], y=list(order), orientation="h",
        marker_color=[COLOURS[name] for name in order], marker_line_width=0,
        text=[pct(v) for v in results.loc[order, "avg_return"]], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
    fig.update_layout(title="Average actual return of each strategy's 10 picks",
                      height=330, margin=dict(l=10, r=60, t=50, b=10),
                      xaxis=dict(tickformat="+.0%"), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        results.reset_index().rename(columns={"index": "Strategy"}),
        hide_index=True, width="stretch",
        column_config={
            "avg_return": st.column_config.NumberColumn("Avg return of picks", format="percent"),
            "vs_market": st.column_config.NumberColumn("vs NIFTY 500", format="percent"),
            "beat_market": st.column_config.NumberColumn("Picks that beat NIFTY 500 (of 10)"),
            "in_actual_top_10pct": st.column_config.NumberColumn(
                "Picks in the actual top 10%"),
            "in_actual_top_10": st.column_config.NumberColumn("Picks in the actual top 10"),
            "best_pick": st.column_config.NumberColumn("Best pick", format="percent"),
            "worst_pick": st.column_config.NumberColumn("Worst pick", format="percent"),
        })

    left, right = st.columns(2)
    with left:
        st.markdown(f"**What {strategy} picked**")
        chosen = trip.picks[trip.picks.strategy == strategy]
        st.dataframe(chosen[["pick", "symbol", "sector", "actual_return", "actual_rank"]],
                     hide_index=True, width="stretch",
                     column_config={"pick": "#", "symbol": "Stock", "sector": "Sector",
                                    "actual_return": st.column_config.NumberColumn(
                                        "Actual return", format="percent"),
                                    "actual_rank": st.column_config.NumberColumn(
                                        f"Actual rank (of {trip.stocks})")})
    with right:
        st.markdown("**What actually did best**")
        st.dataframe(trip.actual_best[["actual_rank", "symbol", "sector", "actual_return"]],
                     hide_index=True, width="stretch",
                     column_config={"actual_rank": "#", "symbol": "Stock", "sector": "Sector",
                                    "actual_return": st.column_config.NumberColumn(
                                        "Actual return", format="percent")})

    st.divider()
    st.markdown("**One date can be lucky or unlucky.** Repeat the test every 3 months over the "
                "whole history for a fair verdict:")
    if st.button(f"Run the time machine on every quarter ({TIME_MACHINE_HORIZONS[tm_horizon]} "
                 "holding, takes 1-2 minutes)"):
        with st.spinner("Travelling to every quarter..."):
            many = travel_many(tm_horizon, refresh_count)
        by_strategy = many.groupby("strategy", sort=False)
        verdict = pd.DataFrame({
            "tests": by_strategy.size(),
            "avg_return": by_strategy.avg_return.mean(),
            "avg_vs_market": by_strategy.vs_market.mean(),
            "won": by_strategy.vs_market.apply(lambda s: (s > 0).mean()),
        })
        st.dataframe(verdict.reset_index(), hide_index=True, width="stretch", column_config={
            "strategy": "Strategy", "tests": "Dates tested",
            "avg_return": st.column_config.NumberColumn("Avg return", format="percent"),
            "avg_vs_market": st.column_config.NumberColumn("Avg vs NIFTY 500", format="percent"),
            "won": st.column_config.NumberColumn("Dates it beat NIFTY 500", format="percent"),
        })

# ---------------------------------------------------------------- sectors

with tabs[3]:
    sectors = analysis.sector_summary(today)
    st.subheader("Which market segments are strongest right now?")
    st.caption(f"Ranked by how the {strategy} strategy scores the sector's stocks, 3-month "
               "performance against the NIFTY 500, and breadth (share of stocks in an uptrend).")

    chart_data = sectors.sort_values("vs_market_3m")
    fig = go.Figure(go.Bar(
        x=chart_data.vs_market_3m, y=chart_data.index, orientation="h",
        marker_color=[BLUE if v >= 0 else RED for v in chart_data.vs_market_3m],
        marker_line_width=0, text=[pct(v) for v in chart_data.vs_market_3m],
        textposition="outside", cliponaxis=False,
        customdata=chart_data[["stocks", "return_3m"]],
        hovertemplate="<b>%{y}</b><br>vs NIFTY 500: %{x:+.1%}<br>"
                      "Median 3-month return: %{customdata[1]:+.1%}<br>"
                      "%{customdata[0]} stocks<extra></extra>"))
    fig.update_layout(title="3-month return vs NIFTY 500 (median stock in each sector)",
                      height=max(380, 26 * len(chart_data)), margin=dict(l=10, r=60, t=50, b=10),
                      xaxis=dict(tickformat="+.0%"), bargap=0.25, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(sectors.reset_index(), hide_index=True, width="stretch", column_config={
        "sector": "Sector", "stocks": "Stocks",
        "return_1m": st.column_config.NumberColumn("1-month return", format="percent"),
        "return_3m": st.column_config.NumberColumn("3-month return", format="percent"),
        "vs_market_3m": st.column_config.NumberColumn("vs NIFTY 500 (3m)", format="percent"),
        "breadth": st.column_config.ProgressColumn(
            "In uptrend", format="percent", min_value=0, max_value=1,
            help="Share of the sector's stocks above their 50-day average"),
        "volatility": st.column_config.NumberColumn("Volatility", format="percent"),
        "avg_rank": None,
        "top_50": st.column_config.NumberColumn(f"Stocks in today's top 50 ({strategy})"),
        "strength": st.column_config.ProgressColumn(
            "Overall strength", format="%.2f", min_value=0, max_value=1),
    })

    chosen_sector = st.selectbox("See the stocks in a sector", sectors.index)
    members = today[today.sector == chosen_sector]
    st.dataframe(members[["name", "price", "signal", "ret_20", "rel_60", "risk_level"]],
                 width="stretch", column_config={
                     "name": "Company",
                     "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                     "signal": "Signal",
                     "ret_20": st.column_config.NumberColumn("1-month return", format="percent"),
                     "rel_60": st.column_config.NumberColumn("vs market (3m)", format="percent"),
                     "risk_level": "Risk"})

# ---------------------------------------------------------------- safety check

with tabs[4]:
    st.subheader("Is this stock safe for me to buy?")
    i1, i2 = st.columns([2, 1])
    symbol = i1.selectbox("Stock", list(today.index),
                          format_func=lambda s: f"{s}  ·  {today.at[s, 'name']}")
    portfolio = i2.number_input("Your total investment money (₹)", min_value=1000,
                                value=100000, step=5000)
    profile = st.radio("How much risk can you handle?", list(PROFILE_LABELS),
                       format_func=PROFILE_LABELS.get, index=1, horizontal=True)

    with st.spinner("Checking the stock..."):
        row = load_stock(symbol, horizon, strategy, refresh_count)
        try:
            stock_news = news.headlines_for(symbol, row["name"], load_headlines())
        except Exception:
            stock_news = pd.DataFrame(columns=["headline", "sentiment"])
    sentiment = stock_news.sentiment.mean() if len(stock_news) else None
    report = analysis.safety_check(row, profile, portfolio, strategy, stats, sentiment)

    show = {"good": st.success, "warning": st.warning, "critical": st.error}[report.tone]
    show(f"**{row['name']}**: {report.verdict}",
         icon={"good": "✅", "warning": "⚠️", "critical": "⛔"}[report.tone])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk level", report.risk_level,
              help=f"Yearly volatility {row.volatility:.0%}, beta {row.beta:.2f}, "
                   f"worst drop in the last year {row.max_drawdown:.0%}")
    m2.metric("Most to put in this stock", rupees(report.suggested_amount),
              help=f"{report.max_allocation:.1%} of your money. Spreading across several "
                   "stocks and sectors limits the damage if one goes wrong.")
    m3.metric("Possible loss in a bad month", rupees(report.bad_month_loss),
              help="On the amount to the left. In about 1 month out of 20 over the last 2 "
                   "years, the stock fell at least this much.")
    m4.metric("Loss if last year's worst drop repeats", rupees(report.worst_year_loss))

    g, b = st.columns(2)
    with g:
        st.markdown("**What looks good**")
        for item in report.positives or ["Nothing stands out"]:
            st.markdown(f"✅ {item}")
    with b:
        st.markdown("**Warning signs**")
        for item in report.warnings or ["None found"]:
            st.markdown(f"⚠️ {item}")

    full = market.close[symbol].dropna()
    history = full.iloc[-504:]
    fig = go.Figure()
    for series, name, colour in [
        (history, "Price", COLOURS[ML]),
        (full.rolling(50).mean().reindex(history.index), "50-day average", COLOURS[MOMENTUM]),
        (full.rolling(200).mean().reindex(history.index), "200-day average", COLOURS[BLEND]),
    ]:
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines",
                                 line=dict(color=colour, width=2), hovertemplate="₹%{y:,.2f}"))
    fig.update_layout(title=f"{symbol}: last 2 years", height=380, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(tickprefix="₹"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")

    d1, d2 = st.columns(2)

    def fmt(value, style):
        return "n/a" if pd.isna(value) else style.format(value)

    with d1:
        st.markdown("**Scores**")
        st.dataframe(pd.DataFrame({
            "": [f"Rank today ({strategy})", "Signal", "Trend checks passed",
                 "Fundamentals checks passed"],
            "Value": [f"top {1 - row.rank_pct:.0%}", row.signal, f"{row.trend_score:.0%}",
                      fmt(row.fundamental_score, "{:.0%}")],
        }), hide_index=True, width="stretch")
    with d2:
        st.markdown("**Company figures** (Yahoo Finance)")
        st.dataframe(pd.DataFrame({
            "": ["Return on equity", "Debt / equity", "P/E", "Profit margin",
                 "Earnings growth (YoY)", "Promoter / insider holding"],
            "Value": [fmt(row.roe, "{:.1%}"), fmt(row.debt_to_equity / 100, "{:.2f}x"),
                      fmt(row.pe, "{:.1f}"), fmt(row.profit_margin, "{:.1%}"),
                      fmt(row.earnings_growth, "{:+.1%}"), fmt(row.promoter_holding, "{:.1%}")],
        }), hide_index=True, width="stretch")

    st.markdown("**Recent news**")
    if len(stock_news):
        for _, item in stock_news.head(8).iterrows():
            mood = "🟢" if item.sentiment > 0 else "🔴" if item.sentiment < 0 else "⚪"
            st.markdown(f"{mood} [{item.headline}]({item.link}) · *{item.source}*")
    else:
        st.caption("No headlines about this company in today's news feeds.")

# ---------------------------------------------------------------- track record

with tabs[5]:
    st.subheader("📒 Live track record")
    st.caption("Every strategy's top-10 list is saved with its date, for every holding period, "
               "whichever one you look at. Once the holding period has passed, it's scored "
               "against the NIFTY 500 (with estimated dividends). Unlike a backtest, these picks "
               "were written down before anyone knew the outcome.")
    record = pipeline.track_record(market)
    if record.empty:
        st.info("No picks recorded yet. They're saved automatically each day you open the app.")
    else:
        finished = record[record.status == "Finished"]
        if len(finished):
            st.dataframe(tracker.summarise(finished), hide_index=True, width="stretch",
                         column_config={
                             "strategy": "Strategy", "horizon": "Days", "lists": "Finished lists",
                             "avg_vs_market": st.column_config.NumberColumn(
                                 "Average vs NIFTY 500", format="percent"),
                             "won": st.column_config.NumberColumn(
                                 "Lists that beat NIFTY 500", format="percent"),
                         })
            st.caption("Lists are saved daily, so their holding periods overlap: 20 finished "
                       "15-day lists are far less evidence than 20 separate periods.")
        no_price = record[record.status == "Missing prices"]
        if len(no_price):
            st.warning(f"{len(no_price)} lists can't be scored: {int(no_price.unpriced.sum())} "
                       "of their picks no longer have any price data. They are left out of the "
                       "summary above, so check those stocks by hand.", icon="⚠️")
        st.dataframe(record, hide_index=True, width="stretch", column_config={
            "unpriced": None,
            "as_of": st.column_config.DateColumn("Picked on"),
            "horizon": "Days", "strategy": "Strategy", "status": "Status",
            "bought": st.column_config.DateColumn("Bought"),
            "until": st.column_config.DateColumn("Held until"),
            "stocks": "Stocks",
            "portfolio_return": st.column_config.NumberColumn("Return", format="percent"),
            "nifty500_return": st.column_config.NumberColumn("NIFTY 500", format="percent"),
            "vs_market": st.column_config.NumberColumn("vs NIFTY 500", format="percent"),
            "beat_market": "Picks beating NIFTY 500",
        })
    st.caption("Stored in `track_record/picks.csv`. Commit it to git regularly so your record "
               "is saved publicly with dates.")

st.divider()
st.caption(DISCLAIMER)
