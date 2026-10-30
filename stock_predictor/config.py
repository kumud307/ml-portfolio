"""Every assumption the project makes, in one place.

Changing a number here changes it everywhere (app, backtest, tests).
"""

# --- What we predict -------------------------------------------------------
HORIZON_DAYS = 15          # hold each pick for 15 trading days (about 3 weeks)
HORIZON_CHOICES = {5: "1 week", 15: "15 trading days", 21: "1 month"}
# Holding periods for the "time machine" test (trading days).
TIME_MACHINE_HORIZONS = {5: "7 days", 21: "1 month", 126: "6 months", 252: "1 year"}
ALL_HORIZONS = tuple(sorted(set(HORIZON_CHOICES) | set(TIME_MACHINE_HORIZONS)))
TOP_N = 10                 # how many stocks the portfolio holds
MAX_PER_SECTOR = 3         # diversification: at most 3 picks from one sector
EXECUTION_LAG_DAYS = 1     # signal seen after today's close -> buy at tomorrow's close
HOLD_BUFFER = 30           # keep a stock we own while it stays in the top 30 (cuts trading costs)

# --- Which stocks are eligible ---------------------------------------------
UNIVERSE_SIZE = 500        # on each date: the 500 most-traded stocks *at that time*
LIQUIDITY_LOOKBACK = 126   # "most traded" = median daily traded value over ~6 months
HISTORY_YEARS = 10         # price history: year 1 warms up the features, the next few
                           # train the first model, so testing includes the 2020 crash
PRICES_FINAL_HOUR_IST = 16 # NSE closes at 15:30; until 16:00 treat today's price as unfinished

# --- Real-world frictions (India, delivery trades) -------------------------
# Round trip = buy + sell: STT 0.1% each side, stamp duty, exchange/SEBI fees,
# GST, DP charge and slippage (the price moving against you while you trade).
COST_ROUND_TRIP = 0.004    # 0.4% of the amount traded
SHORT_TERM_TAX = 0.20      # STCG tax on profits from shares held under 1 year
RISK_FREE_RATE = 0.065     # roughly what a bank FD / T-bill pays; used in Sharpe ratio
# Stock prices include dividends but the NIFTY 500 index level doesn't, so this
# estimate of its dividend yield is added back whenever the two are compared.
INDEX_DIVIDEND_YIELD = 0.01
