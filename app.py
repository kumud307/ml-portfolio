"""NSE Stock Analyser - Streamlit dashboard.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_predictor import analysis, news, pipeline
from stock_predictor.universe import BENCHMARK

st.set_page_config(page_title="NSE Stock Analyser", page_icon="📈", layout="wide")

# Chart colours (validated colour-blind-safe reference palette).
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

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


# ---------------------------------------------------------------- data loading

@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def load_market(refresh_count):
    return pipeline.run(refresh=refresh_count > 0)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_top_picks(as_of, refresh_count):
    return pipeline.top_picks(load_market(refresh_count))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_stock(symbol, as_of, refresh_count):
    return pipeline.stock_details(load_market(refresh_count), symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def load_headlines():
    return news.fetch_headlines()


st.session_state.setdefault("refresh_count", 0)

with st.sidebar:
    st.header("📈 NSE Stock Analyser")
    st.caption("Market segments, stock ranking and a personal risk check for "
               "the NIFTY 500 (about 95% of NSE trading).")
    if st.button("🔄 Refresh data now", width="stretch"):
        st.session_state.refresh_count += 1
        st.cache_data.clear()
    st.caption("Prices refresh automatically every 6 hours; company figures weekly.")
    st.divider()
    st.warning(DISCLAIMER, icon="⚠️")

refresh_count = st.session_state.refresh_count
with st.spinner("Downloading prices and training the model (first run takes about a minute)..."):
    try:
        market = load_market(refresh_count)
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

table = market.table
as_of = str(market.as_of.date())
benchmark = market.close[BENCHMARK]


def pct(value, signed=True):
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def rupees(value):
    return f"₹{value:,.0f}"


# ---------------------------------------------------------------- header

st.title("NSE Stock Analyser")
st.caption(f"Prices as of **{as_of}** · {len(table)} stocks analysed · "
           f"predictions look {market.horizon} trading days (about 1 month) ahead")

c1, c2, c3, c4 = st.columns(4)
c1.metric("NIFTY 500: last month", pct(benchmark.iloc[-1] / benchmark.iloc[-21] - 1))
c2.metric("NIFTY 500: last 3 months", pct(benchmark.iloc[-1] / benchmark.iloc[-61] - 1))
c3.metric("Stocks above their 50-day average", pct((table.sma50_gap > 0).mean(), signed=False),
          help="Market breadth. Above 50% means most stocks are in short-term uptrends.")
c4.metric("Model accuracy (AUC)", f"{market.backtest.auc:.2f}",
          help="0.50 = coin flip, 1.00 = perfect. See the 'How accurate is this?' tab.")

tab_sectors, tab_picks, tab_check, tab_model = st.tabs(
    ["🏭 Market segments", "🏆 Top stock picks", "🛡️ Is it safe for me?", "🎯 How accurate is this?"])


# ---------------------------------------------------------------- sectors

with tab_sectors:
    sectors = analysis.sector_summary(table)
    st.subheader("Which market segments are strongest right now?")
    st.caption("Segments are ranked by a mix of the model's outlook, 3-month performance against "
               "the NIFTY 500, and breadth (share of stocks in an uptrend).")

    chart_data = sectors.sort_values("vs_market_3m")
    fig = go.Figure(go.Bar(
        x=chart_data.vs_market_3m,
        y=chart_data.index,
        orientation="h",
        marker_color=[BLUE if v >= 0 else RED for v in chart_data.vs_market_3m],
        marker_line_width=0,
        text=[pct(v) for v in chart_data.vs_market_3m],
        textposition="outside",
        cliponaxis=False,
        customdata=chart_data[["stocks", "return_3m"]],
        hovertemplate="<b>%{y}</b><br>vs NIFTY 500: %{x:+.1%}<br>"
                      "Median 3-month return: %{customdata[1]:+.1%}<br>"
                      "%{customdata[0]} stocks<extra></extra>",
    ))
    fig.update_layout(
        title="3-month return vs NIFTY 500 (median stock in each segment)",
        height=max(380, 26 * len(chart_data)),
        margin=dict(l=10, r=60, t=50, b=10),
        xaxis=dict(tickformat="+.0%", zeroline=True, zerolinewidth=1),
        bargap=0.25,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        sectors.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={
            "sector": "Segment",
            "stocks": "Stocks",
            "return_1m": st.column_config.NumberColumn("1-month return", format="percent"),
            "return_3m": st.column_config.NumberColumn("3-month return", format="percent"),
            "vs_market_3m": st.column_config.NumberColumn("vs NIFTY 500 (3m)", format="percent"),
            "breadth": st.column_config.ProgressColumn(
                "In uptrend", format="percent", min_value=0, max_value=1,
                help="Share of the segment's stocks above their 50-day average"),
            "volatility": st.column_config.NumberColumn(
                "Volatility", format="percent", help="Typical yearly price swing"),
            "ml_outperform_prob": st.column_config.NumberColumn(
                "Model: chance to beat market", format="percent"),
            "trend_score": None,
            "strength": st.column_config.ProgressColumn(
                "Overall strength", format="%.2f", min_value=0, max_value=1),
        },
    )

    chosen = st.selectbox("See the stocks in a segment", sectors.index)
    members = table[table.sector == chosen].sort_values("shortlist_score", ascending=False)
    st.dataframe(
        members[["name", "price", "ret_20", "rel_60", "ml_prob", "trend_score", "risk_level"]],
        width="stretch",
        column_config={
            "name": "Company",
            "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
            "ret_20": st.column_config.NumberColumn("1-month return", format="percent"),
            "rel_60": st.column_config.NumberColumn("vs market (3m)", format="percent"),
            "ml_prob": st.column_config.NumberColumn("Model: chance to beat market", format="percent"),
            "trend_score": st.column_config.ProgressColumn("Trend", min_value=0, max_value=1, format="%.2f"),
            "risk_level": "Risk",
        },
    )


# ---------------------------------------------------------------- top picks

with tab_picks:
    st.subheader("Best-ranked stocks right now")
    st.caption(
        "How the ranking works: every stock gets a **model score** (chance of beating the NIFTY 500 "
        "next month) and a **trend score** (moving averages, momentum, RSI). The best 50 are then "
        "checked on **company fundamentals** (ROE, debt, P/E vs sector, margins, earnings growth). "
        "Final score = 45% model + 25% trend + 30% fundamentals.")

    with st.spinner("Fetching company fundamentals for the shortlist..."):
        picks = load_top_picks(as_of, refresh_count)

    f1, f2, f3 = st.columns([2, 1, 1])
    sector_filter = f1.multiselect("Segments", sorted(picks.sector.unique()), placeholder="All segments")
    max_risk = f2.selectbox("Maximum risk", ["High", "Medium", "Low"])
    count = f3.slider("Show top", 5, 50, 20)

    allowed_risk = {"Low": ["Low"], "Medium": ["Low", "Medium"], "High": ["Low", "Medium", "High"]}[max_risk]
    shown = picks[picks.risk_level.isin(allowed_risk)]
    if sector_filter:
        shown = shown[shown.sector.isin(sector_filter)]
    shown = shown.head(count)

    st.dataframe(
        shown[["name", "sector", "price", "signal", "score", "ml_prob", "trend_score",
               "fundamental_score", "risk_level", "rel_60", "volatility", "pe", "roe"]],
        width="stretch",
        column_config={
            "name": "Company",
            "sector": "Segment",
            "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
            "signal": "Signal",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
            "ml_prob": st.column_config.NumberColumn("Model: chance to beat market", format="percent"),
            "trend_score": st.column_config.NumberColumn("Trend", format="%.2f"),
            "fundamental_score": st.column_config.NumberColumn("Fundamentals", format="%.2f"),
            "risk_level": "Risk",
            "rel_60": st.column_config.NumberColumn("vs market (3m)", format="percent"),
            "volatility": st.column_config.NumberColumn("Volatility", format="percent"),
            "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
            "roe": st.column_config.NumberColumn("ROE", format="percent"),
        },
    )
    if shown.empty:
        st.info("No stocks match these filters. Try allowing more risk or more segments.")
    st.download_button("Download this list (CSV)", shown.to_csv().encode("utf-8"),
                       file_name=f"top_picks_{as_of}.csv", mime="text/csv")
    st.caption("Use **Is it safe for me?** to check a stock against your own budget and risk tolerance.")


# ---------------------------------------------------------------- safety check

with tab_check:
    st.subheader("Is this stock safe for me to buy?")
    options = list(load_top_picks(as_of, refresh_count).index) + sorted(
        set(table.index) - set(load_top_picks(as_of, refresh_count).index))
    i1, i2 = st.columns([2, 1])
    symbol = i1.selectbox("Stock", options, format_func=lambda s: f"{s}  ·  {table.at[s, 'name']}")
    portfolio = i2.number_input("Your total investment money (₹)", min_value=1000, value=100000, step=5000)
    profile = st.radio("How much risk can you handle?", list(PROFILE_LABELS),
                       format_func=PROFILE_LABELS.get, index=1, horizontal=True)

    with st.spinner("Checking the stock..."):
        row = load_stock(symbol, as_of, refresh_count)
        try:
            stock_news = news.headlines_for(symbol, row["name"], load_headlines())
        except Exception:
            stock_news = pd.DataFrame(columns=["headline", "sentiment"])
    sentiment = stock_news.sentiment.mean() if len(stock_news) else None
    report = analysis.safety_check(row, profile, portfolio, market.backtest, sentiment)

    show = {"good": st.success, "warning": st.warning, "critical": st.error}[report.tone]
    show(f"**{row['name']}**: {report.verdict}",
         icon={"good": "✅", "warning": "⚠️", "critical": "⛔"}[report.tone])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk level", report.risk_level,
              help=f"Yearly volatility {row.volatility:.0%}, beta {row.beta:.2f}, "
                   f"worst drop in the last year {row.max_drawdown:.0%}")
    m2.metric("Most to put in this stock", rupees(report.suggested_amount),
              help=f"{report.max_allocation:.1%} of your money. Spreading across several "
                   "stocks and segments limits the damage if one goes wrong.")
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

    history = market.close[symbol].dropna().iloc[-504:]
    full = market.close[symbol].dropna()
    fig = go.Figure()
    for series, name, colour in [
        (history, "Price", BLUE),
        (full.rolling(50).mean().reindex(history.index), "50-day average", ORANGE),
        (full.rolling(200).mean().reindex(history.index), "200-day average", AQUA),
    ]:
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines",
                                 line=dict(color=colour, width=2),
                                 hovertemplate="₹%{y:,.2f}"))
    fig.update_layout(
        title=f"{symbol}: last 2 years",
        height=380,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(tickprefix="₹"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Scores**")
        st.dataframe(pd.DataFrame({
            "": ["Overall score", "Model: chance to beat market", "Trend score",
                 "Fundamentals score", "Signal"],
            "Value": [f"{row.score:.2f}", f"{row.ml_prob:.0%}", f"{row.trend_score:.2f}",
                      "n/a" if pd.isna(row.fundamental_score) else f"{row.fundamental_score:.2f}",
                      row.signal],
        }), hide_index=True, width="stretch")
    with d2:
        st.markdown("**Company figures** (Yahoo Finance)")

        def fmt(value, style):
            return "n/a" if pd.isna(value) else style.format(value)

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


# ---------------------------------------------------------------- model accuracy

with tab_model:
    bt = market.backtest
    st.subheader("How accurate is the prediction model?")
    st.caption(
        f"The model was repeatedly trained on past data and tested on the following months "
        f"it had never seen ({bt.test_from:%b %Y} to {bt.test_to:%b %Y}). For each stock and "
        f"day it predicted: will this stock beat the NIFTY 500 over the next "
        f"{market.horizon} trading days?")

    a1, a2, a3 = st.columns(3)
    a1.metric("AUC", f"{bt.auc:.3f}", help="0.50 = coin flip, 1.00 = perfect")
    a2.metric("Correct predictions", pct(bt.accuracy, signed=False),
              help=f"For comparison, stocks beat the market {bt.base_rate:.0%} of the time")
    a3.metric("Model's top-20% picks vs average stock (per month)",
              pct(bt.top_pick_excess - bt.avg_excess))

    if bt.has_edge:
        st.success("The model showed a small but real edge in testing. Treat it as one input, not a guarantee.")
    else:
        st.warning(
            "**The model's edge is weak, close to a coin flip.** Short-term stock prices are "
            "mostly random, and professional funds also struggle to predict them. That's why the "
            "final score also uses trend and company fundamentals, and why the safety check "
            "focuses on **risk**: how much you could lose and how much to invest. Risk can be "
            "measured far more reliably than future returns.")

    st.markdown("""
**What goes into the model** (15 signals per stock, per day): returns over 1 week to 6 months ·
performance against the NIFTY 500 and against its own segment · volatility · distance from the
50- and 200-day averages · RSI · distance from the 1-year high · trading-volume trend.

**How risk is rated:** yearly volatility, beta (how much it moves with the market) and the
worst fall in the last year are combined into Low / Medium / High.

**Suggested position size:** at most 5% (low risk tolerance), 10% (medium) or 15% (high) of
your money in a single stock, halved for High-risk stocks.

**Data:** prices and company figures from Yahoo Finance, the stock list from NSE, and news from
Moneycontrol, Economic Times, Mint, Business Standard, Financial Express and NDTV Profit.
""")
    if market.missing:
        st.caption(f"{len(market.missing)} NIFTY 500 stocks were skipped for having under a year "
                   f"of price history: {', '.join(market.missing)}")

st.divider()
st.caption(DISCLAIMER)
