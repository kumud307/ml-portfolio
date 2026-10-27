# NSE Stock Analyser

A Streamlit dashboard that analyses the Indian stock market (NIFTY 500, about 95% of NSE
trading). It compares market segments, ranks stocks using machine learning together with
trend and company fundamentals, and checks whether a stock suits **your** budget and risk
tolerance.

> **Not financial advice.** This is an educational project. Its predictions are rough
> probabilities, and the backtest below shows how limited they are.

## Features

| Tab | What it does |
|---|---|
| 🏭 **Market segments** | Ranks 20 sectors (IT, banking, pharma, auto, ...) by 3-month performance vs the NIFTY 500, breadth (share of stocks in an uptrend), volatility and the model's outlook. |
| 🏆 **Top stock picks** | Scores all ~480 stocks, shortlists the best 50, then adds fundamentals (ROE, debt, P/E vs sector, margins, earnings growth). Filter by sector and risk level, then download as CSV. |
| 🛡️ **Is it safe for me?** | Pick a stock, enter your budget and risk tolerance. Returns a verdict, risk level, suggested maximum position, estimated rupee loss in a bad month, red flags, a price chart and recent news. |
| 🎯 **How accurate is this?** | Walk-forward backtest of the model on data it never saw. |

## How it works

1. **Data:** stock list and sectors from NSE; 5 years of daily prices and company figures
   from Yahoo Finance (`yfinance`); headlines from Indian financial news RSS feeds.
2. **Features (15 per stock per day):** returns over 1 week to 6 months, performance vs the
   index and vs its own sector, volatility, distance from the 50/200-day averages, RSI,
   drawdown from the 1-year high, volume trend.
3. **Model:** `HistGradientBoostingClassifier` predicting whether a stock beats the NIFTY 500
   over the next 20 trading days.
4. **Validation:** walk-forward. The model is trained on the past and tested on the following
   unseen months, with a 20-day gap to prevent look-ahead leakage.
5. **Final score:** 45% model, 25% trend, 30% fundamentals.
6. **Risk:** volatility, beta and maximum drawdown give a Low / Medium / High rating. Position
   size is capped at 5/10/15% of your money (by risk tolerance), halved for High-risk stocks.

### Honest results

Out-of-sample AUC is about **0.51** (0.50 = coin flip), and the model's top-20% picks beat the
average stock by roughly **0.4% per month**. Short-term returns are mostly noise, so the app
treats the model as one weak input and puts more weight on **risk management**, which can be
measured reliably.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

In VS Code, press **F5** and choose **Run dashboard (Streamlit)**. The first run downloads
data and trains the model, which takes about a minute. After that, data is cached in `data/`.

## Project structure

```
app.py                      Streamlit dashboard
stock_predictor/
  universe.py               NIFTY 500 stock list + sectors (from NSE)
  data.py                   price and fundamentals download, with caching
  features.py               technical features and prediction target
  model.py                  gradient boosting model + walk-forward backtest
  analysis.py               risk metrics, scoring, sectors, safety check
  news.py                   news headlines + simple sentiment
  pipeline.py               ties it all together
assets/nifty500_list.csv    fallback copy of the NSE stock list
```
