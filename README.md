# NSE Stock Analyser

A Streamlit app that picks a **top-10 list of NSE stocks for the next 15 days** and checks
every idea against history before trusting it. Strategies are compared on a walk-forward
backtest after trading costs and tax. A "time machine" replays any past date, and a live
track record scores each list once its outcome is known.

> **Not financial advice.** Educational project. The research notes show how limited
> short-term prediction is.

📄 **[Research notes: method, every experiment, results](RESEARCH.md)**

## Headline results (15-day holding, Jul 2018 to Sep 2026, after costs)

| Strategy | Yearly return | Sharpe | IC |
|---|---|---|---|
| ML model (gradient boosting ranker) | 18.0% | 0.54 | 0.034 (t = 2.8) |
| Momentum (12-1 month) | 39.7% | 1.02 | 0.045 |
| Average stock (equal weight) | 21.5% | 0.68 | |
| NIFTY 500 | 11.0% | 0.32 | |

The ML signal is statistically real but doesn't beat equal-weighting after costs; momentum
leads but is the most flattered by the remaining survivorship bias. Details and caveats are in
[RESEARCH.md](RESEARCH.md).

## Features

| Tab | What it does |
|---|---|
| 🏆 **Top 10 picks** | Today's 10 picks (max 3 per sector) for a 1-week, 15-day or 1-month hold, with shares to buy for your budget, risk level and company-quality checks |
| 📊 **Backtest** | 5 strategies vs 2 benchmarks: CAGR, after-tax CAGR, Sharpe, drawdown, hit rate, turnover, IC, and a growth chart |
| ⏳ **Time machine** | Go back to any date, train only on what was known then, pick 10 stocks for 7 days / 1 month / 6 months / 1 year, and compare with what actually happened, for one date or every quarter |
| 🏭 **Market segments** | Sector strength, breadth and momentum |
| 🛡️ **Is it safe for me?** | Risk level, position-size limit, rupee loss in a bad month, warning signs, chart and news for any stock |
| 📒 **Live track record** | Every list the app shows is saved and later scored vs the NIFTY 500 |

## How it works

1. **Universe:** about 755 NSE stocks; on each date only the 500 most-traded *at that time* are
   eligible (reduces survivorship bias).
2. **Features:** 17 per-stock signals ranked within each day, plus 7 market-regime signals
   (India VIX, market trend, breadth).
3. **Model:** a gradient-boosting regressor predicting each stock's rank for the next 15 days.
4. **Validation:** walk-forward with a purge gap, so the model never trains on answers it
   couldn't have known. Unit tests check this.
5. **Simulation:** buy the day after the signal, 0.4% round-trip costs, 20% STCG tax, hold
   buffer to cut turnover.

All assumptions live in [`stock_predictor/config.py`](stock_predictor/config.py).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

In VS Code, press **F5** → **Run dashboard (Streamlit)**. The first run downloads 10 years of
prices for about 755 stocks and trains the models, which takes 2-3 minutes; after that,
results are cached in `data/` for the day.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a synthetic market (no internet). They check that features never use future
prices, that training never sees unknown answers, that a "perfect foresight" score wins and a
random one doesn't, and that costs, position sizing and the track record are correct.

## Project structure

```
app.py                       Streamlit dashboard
stock_predictor/
  config.py                  every assumption: horizon, costs, tax, universe size
  universe.py                NSE stock list + point-in-time "most traded" filter
  data.py                    prices, India VIX and fundamentals (Yahoo Finance), cached
  features.py                features, regime signals and forward-return targets
  model.py                   gradient-boosting ranker + walk-forward training
  backtest.py                top-10 portfolio simulation, strategies, metrics
  time_machine.py            replay a past date and compare with reality
  tracker.py                 live track record
  analysis.py                risk, sectors, fundamentals checks, safety verdict
  news.py                    news headlines + simple sentiment
  pipeline.py                ties everything together, with caching
tests/                       pytest suite (synthetic data)
track_record/picks.csv       saved picks (commit this to keep the live record)
RESEARCH.md                  method, experiments, results, limitations
```
