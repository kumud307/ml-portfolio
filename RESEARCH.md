# Research notes: can we pick the 10 best NSE stocks for the next 15 days?

**Short answer:** not reliably. The ML model finds a small but statistically significant
signal (IC 0.034, t = 2.8), but after trading costs it earns less than simply holding every
stock equally. Classic 12-1 month momentum did best in the backtest, but it is also the
strategy most flattered by the remaining survivorship bias, and it crashed in early 2025.

All numbers below come from `stock_predictor/` and can be reproduced from the app.

---

## 1. Question

Every 15 trading days, choose 10 NSE stocks to hold for the next 15 days. Can a machine
learning model do better, after costs and tax, than (a) the NIFTY 500 index, (b) holding every
stock equally, and (c) simple rules such as momentum?

## 2. Data and universe

| Item | Choice | Why |
|---|---|---|
| Candidates | NSE NIFTY Total Market list (755 stocks with sectors) | broad market, sector labels from NSE |
| Eligible on each date | the 500 most-traded stocks by 6-month median traded value **as of that date** | avoids using today's index membership in the past |
| History | 10 years of daily prices (Yahoo Finance), India VIX | the first year warms up features; testing covers the 2020 crash |
| Benchmark | NIFTY 500 index (`^CRSLDX`) | |

**Known bias:** companies delisted since then (bankruptcies, mergers) are not in today's NSE
list, so they're missing from the history. The liquidity filter fixes "chosen with
hindsight", but not "missing the losers". All absolute returns are therefore overstated, and
momentum most of all, because it buys stocks after big moves.

## 3. Method

- **Features (24):** 17 per-stock signals (returns over 1 week to 6 months, 12-1 momentum,
  return vs market and vs own sector, volatility, distance from 50/200-day averages, RSI,
  drawdown, volume trend, liquidity), each converted to a **percentile rank within the day**,
  plus 7 market-regime signals (market return and trend, market volatility, India VIX and
  its change, breadth).
- **Target:** the percentile rank of each stock's next-15-day return in excess of the
  NIFTY 500.
- **Model:** `HistGradientBoostingRegressor` with heavy regularisation (min 500 samples per
  leaf), trained on every 5th day, because neighbouring days' targets overlap.
- **Validation:** expanding-window walk-forward in 12 blocks. Each block is predicted by a
  model trained only on rows whose targets were complete before the block began (a gap of
  15 + 1 days). There is a unit test for this rule (`tests/test_model.py`).
- **Execution:** signal at day *t*'s close, buy at *t+1*'s close, hold 15 days.
- **Portfolio:** top 10 by score, equal weight, max 3 per sector; a held stock is kept while it
  stays in the top 30 (the "hold buffer").
- **Costs:** 0.4% per round trip on the share of the portfolio that changes (STT, stamp duty,
  exchange fees, GST, slippage). **Tax:** 20% STCG on each calendar year's profit.
- **Metrics:** CAGR, Sharpe ratio (vs a 6.5% risk-free rate), maximum drawdown, hit rate vs
  the NIFTY 500, turnover, and the Information Coefficient (per-date Spearman correlation
  between score and realised excess return).

## 4. Experiment log

Every variant tried is listed, including the rejected ones, because trying many ideas and
reporting only the best is a hidden form of overfitting.

| # | Change | Result | Kept? |
|---|---|---|---|
| v1 | Classifier "beats the index in 20 days?", fixed NIFTY 500 list, 5 years | AUC 0.51. Top 10 over 15 days: +0.76% vs index per period, **below** both the average stock (+0.87%) and 3-month momentum (+2.34%) | ❌ replaced |
| v2 | Ranking regressor, rank-normalised features, regime features, point-in-time universe, 10 years, 1-day execution lag, costs | ML CAGR 16.8%, IC 0.03, but turnover 85% per period | base |
| E1 | + hold buffer (keep while in the top 30) | ML CAGR 16.8% → 18.3%; turnover 85% → 70%; momentum turnover 32% → 15% | ✅ |
| E2 | Train on a 45-day target, trade every 15 days (less noisy target) | ML CAGR 18.3% → 14.0%, worse | ❌ |

## 5. Results: 15-day holding, July 2018 to September 2026 (135 periods)

After costs; "after tax" also applies 20% STCG yearly.

| Strategy | CAGR | After tax | Sharpe | Max drawdown | Beat NIFTY 500 | Turnover | IC (t-stat) |
|---|---|---|---|---|---|---|---|
| ML model | 18.0% | 14.4% | 0.54 | −36.9% | 56% | 70% | 0.034 (2.8) |
| ML + Momentum | 25.2% | 20.3% | 0.75 | −29.7% | 57% | 53% | 0.046 (3.5) |
| **Momentum (12-1)** | **39.7%** | **31.8%** | **1.02** | −37.6% | 61% | 15% | 0.045 (3.2) |
| Momentum (3 months) | 23.7% | 18.6% | 0.61 | −61.2% | 56% | 37% | 0.010 (0.9) |
| Reversal (worst week) | 18.6% | 14.5% | 0.48 | −43.4% | 52% | 90% | 0.012 (1.3) |
| Average stock | 21.5% | (taxed only on sale) | 0.68 | −40.0% | 60% | 0% | |
| NIFTY 500 | 11.0% | (taxed only on sale) | 0.32 | −30.6% | | 0% | |

With a 1-week holding period the ML IC rises to 0.044 (t = 6.7, 407 periods), but more
frequent trading means its CAGR is still only 18.8%.

## 6. Time-machine check: one test every quarter, no costs

Train only on what was known on the date, pick 10, compare with what actually happened.

| Holding | Dates tested | ML vs NIFTY 500 (avg) | ML won | Average stock vs NIFTY 500 | Momentum 12-1 vs NIFTY 500 |
|---|---|---|---|---|---|
| 7 days | 32 | +0.2% | 47% | −0.1% | +0.6% |
| 1 month | 32 | +1.4% | 72% | +1.2% | +2.6% |
| 6 months | 28 | +10.5% | 79% | +8.1% | +17.9% |
| 1 year | 24 | +12.9% | 79% | +21.3% | +44.6% |

**Case study, 31 December 2024:** every strategy did badly. Over 1 year, the ML model's 10
picks returned −23% while the NIFTY 500 returned +3% (1 of 10 picks beat the index). Early
2025's small/mid-cap correction hit exactly the stocks that had run hardest in 2024. A single
date says little; it shows how large the risk of any one list is.

## 7. Conclusions

1. **The ML signal is real but small.** IC 0.03-0.04 is typical of genuine equity signals and
   is statistically significant here. But its picks change a lot, and after costs it doesn't
   beat equal-weighting the whole universe (18.0% vs 21.5%).
2. **Most "beating the market" here isn't skill.** The equal-weighted average stock beat the
   cap-weighted NIFTY 500 in most periods (small/mid caps outperformed in 2019-24), helped by
   survivorship bias. The fair yardstick is the average stock, not the index.
3. **12-1 momentum was the strongest and cheapest to trade** (15% turnover), consistent with
   decades of research. But its result is the most inflated by the missing delisted stocks,
   and its drawdowns are deep (−38%).
4. **Choosing the best of five strategies after seeing the results is optimistic.** Expect
   future returns to be lower than the table.
5. **For a real investor,** the defensible use of this tool is a disciplined, diversified,
   momentum-tilted list with strict position sizing, run on paper first. The live track record
   (`track_record/picks.csv`) is the real test.

## 8. Limitations and next steps

- **Delisted stocks:** add NSE's historical delisting data to remove the remaining survivorship
  bias. This is the most important fix.
- **Better costs:** a slippage model that depends on liquidity (small caps cost more to trade).
- **India-specific data:** delivery % (NSE bhavcopy), FII/DII flows, and earnings-announcement
  dates.
- **Model:** a learning-to-rank objective (LightGBM LambdaRank), a turnover penalty in
  training, and regime-aware switching between momentum and reversal.
- **Statistics:** bootstrap confidence intervals on CAGR and Sharpe; a correction for testing
  several strategies.
