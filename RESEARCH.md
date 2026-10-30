# Research notes: can we pick the 10 best NSE stocks for the next 15 days?

**Short answer:** not reliably. The ML model finds a small but statistically significant
signal (IC 0.034, t = 2.8), but after trading costs its typical result is below simply holding
every stock equally, and its return swings widely on arbitrary choices such as the day the
cycle starts. Classic 12-1 month momentum did best on every start day, but survivorship bias
inflates it heavily, and it crashed in early 2025.

Sections 5 and 6 come from `stock_predictor/` and use prices downloaded on 28 October 2026, up
to the last finished day (27 October). The app reproduces them from the same prices; a later
download changes them, as section 5 explains. The log in section 4 records each experiment's
result at the time, against a benchmark without dividends.

---

## 1. Question

Every 15 trading days, choose 10 NSE stocks to hold for the next 15 days. Can a machine
learning model do better, after costs and tax, than (a) the NIFTY 500 index, (b) holding every
stock equally, and (c) simple rules such as momentum?

## 2. Data and universe

| Item | Choice | Why |
|---|---|---|
| Candidates | NSE NIFTY Total Market list as it is today (755 stocks with sectors, 714 with enough price history) | broad market, sector labels from NSE |
| Eligible on each date | the 500 most-traded stocks by 6-month median traded value **as of that date** | a stock only counts once it is actively traded |
| History | 10 years of daily prices (Yahoo Finance), India VIX | the first year warms up features; testing covers the 2020 crash |
| Benchmark | NIFTY 500 index (`^CRSLDX`) plus an estimated 1% a year of dividends | stock prices include dividends but the index level doesn't; the index's dividend yield is about 1% |

**Known bias: survivorship.** Every candidate is on today's list, so the history only contains
companies that survived and stayed big enough. Companies that collapsed, were delisted or
simply shrank off the list are missing, and small companies that later grew are included from
the start.

The liquidity filter can't fix this. It only chooses among today's survivors, and in the early
years it hardly chooses at all, because so few of them have price data:

| Year | Candidates with data | Left out by the filter |
|---|---|---|
| 2018 | 485 | 0% |
| 2019 | 504 | 1% |
| 2020 | 522 | 4% |
| 2021 | 546 | 8% |
| 2022 | 586 | 15% |
| 2023 | 610 | 18% |
| 2024 | 643 | 22% |
| 2025 | 685 | 27% |
| 2026 | 714 | 30% |

All absolute returns are therefore overstated, and momentum most of all, because it buys stocks
after big moves. Section 5 shows how much the results depend on the smaller stocks.

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
- **Execution:** signal at day *t*'s close, buy at *t+1*'s close, hold 15 days. Live lists use
  finished days only: until 16:00 IST, today's unfinished price is ignored.
- **Portfolio:** top 10 by score, equal weight, max 3 per sector; a held stock is kept while it
  stays in the top 30 (the "hold buffer").
- **Costs:** 0.4% per round trip on the share of the portfolio that changes (STT, stamp duty,
  exchange fees, GST, slippage). **Tax:** 20% STCG on each calendar year's profit, for the
  strategies only, because the benchmarks are taxed only when finally sold.
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
| ML model | 16.8% | 13.5% | 0.50 | −36.9% | 55% | 70% | 0.034 (2.8) |
| ML + Momentum | 27.5% | 22.1% | 0.82 | −29.8% | 58% | 52% | 0.046 (3.5) |
| **Momentum (12-1)** | **39.7%** | **31.8%** | **1.02** | −37.6% | 61% | 15% | 0.045 (3.2) |
| Momentum (3 months) | 23.7% | 18.6% | 0.61 | −61.2% | 56% | 37% | 0.010 (0.9) |
| Reversal (worst week) | 18.6% | 14.5% | 0.48 | −43.4% | 51% | 90% | 0.012 (1.3) |
| Average stock | 21.5% | (taxed only on sale) | 0.68 | −40.0% | 58% | 0% | |
| NIFTY 500, with estimated dividends | 12.1% | (taxed only on sale) | 0.38 | −30.5% | | 0% | |

With a 1-week holding period the ML IC rises to 0.044 (t = 6.6, 407 periods), but more
frequent trading means its CAGR is still only 18.3%.

### How far can these numbers be trusted?

**The start day matters, especially for ML.** A 15-day cycle has 15 possible start days. Using
the same data and the same model scores, only the start day changes here:

| Strategy | Published above | Lowest | Median | Highest |
|---|---|---|---|---|
| ML model | 16.8% | 11.2% | 18.5% | 29.0% |
| ML + Momentum | 27.5% | 22.1% | 31.1% | 38.2% |
| Momentum (12-1) | 39.7% | 34.4% | 37.1% | 41.2% |
| Momentum (3 months) | 23.7% | 18.4% | 24.6% | 35.1% |
| Reversal (worst week) | 18.6% | 0.4% | 14.1% | 28.4% |
| Average stock | 21.5% | 21.3% | 22.0% | 22.9% |
| NIFTY 500, with estimated dividends | 12.1% | 11.8% | 12.2% | 12.6% |

Momentum came first on all 15 days. ML beat the average stock on only 4 of them. Its IC stays
between 0.03 and 0.05 throughout, so the signal is stable, but the 10 stocks it picks aren't.
Ending the history one to six days earlier, which moves the retraining dates slightly, also
moves ML between 11.8% and 16.8%. Any single ML return here should be read as very approximate.

**The results depend heavily on smaller stocks.** Survivorship bias matters most among them, so
this shows the same backtest on only the most-traded stocks each day:

| Stocks included each day | Momentum | ML model | Average stock |
|---|---|---|---|
| All eligible (about 500) | 39.7% | 16.8% | 21.5% |
| Top ~200 by traded value | 32.3% | 15.7% | 16.1% |
| Top ~100 by traded value | 21.8% | 11.9% | 13.8% |

The NIFTY 500 returned 12.1% over the same period. Part of the fall is real small-cap
outperformance, so this doesn't measure the bias exactly. Momentum still beat the average stock
at every size, while ML fell slightly below the index on the top ~100.

**A few trades carry momentum.** Without its best 14 of 1,350 positions, momentum's return
before costs drops from 41.1% to 27.7%. Removing only the winners is a harsh test, but it shows
how much of the result comes from a few stocks. It also made more than 110% in each of 2020 and
2021.

## 6. Time-machine check: one test every quarter, no costs

Train only on what was known on the date, pick 10, compare with what actually happened.

| Holding | Dates tested | ML vs NIFTY 500 (avg) | ML won | Average stock vs NIFTY 500 | Momentum 12-1 vs NIFTY 500 |
|---|---|---|---|---|---|
| 7 days | 32 | +0.2% | 50% | −0.1% | +0.6% |
| 1 month | 32 | +1.4% | 63% | +1.1% | +2.5% |
| 6 months | 28 | +10.0% | 75% | +7.6% | +17.4% |
| 1 year | 24 | +11.8% | 75% | +20.1% | +43.4% |

**Case study, 31 December 2024:** every strategy did badly. Over 1 year, the ML model's 10
picks returned −18% while the NIFTY 500 returned +4% (2 of 10 picks beat the index). Early
2025's small/mid-cap correction hit exactly the stocks that had run hardest in 2024. A single
date says little; it shows how large the risk of any one list is.

## 7. Conclusions

1. **The ML signal is real, but its return is unreliable.** IC 0.03-0.05 is typical of genuine
   equity signals and is statistically significant here. But its top 10 is unstable: its return
   runs from 11.2% to 29.0% across start days, and the median (18.5%) is below equal-weighting
   the whole universe (22.0%).
2. **Most "beating the market" here isn't skill.** The equal-weighted average stock beat the
   cap-weighted NIFTY 500 in most periods (small/mid caps outperformed in 2019-24), helped
   heavily by survivorship bias. On the 100 most-traded stocks, its lead shrinks from 9.4 points
   to 1.7. The fair yardstick is the average stock, not the index.
3. **12-1 momentum was the strongest and cheapest to trade** (15% turnover), consistent with
   decades of research. It came first on all 15 start days and beat the average stock at every
   universe size. But its headline is the most inflated by the missing stocks (21.8% on the 100
   most-traded), a few trades carry much of it, and its drawdowns are deep (−38%).
4. **Choosing the best of five strategies after seeing the results is optimistic.** Expect
   future returns to be lower than the table.
5. **For a real investor,** the defensible use of this tool is a disciplined, diversified,
   momentum-tilted list with strict position sizing, run on paper first. The live track record
   (`track_record/picks.csv`) is the real test.

## 8. Limitations and next steps

- **Survivorship bias:** use historical index membership (which stocks were really in the index
  on each past date), plus prices for stocks that have since left it. Delisting data alone isn't
  enough, because stocks that only shrank are missing too. This is the most important fix.
- **Measure the bias directly:** compare the average stock with the published NIFTY 500 Equal
  Weight index over the same dates.
- **Real benchmark:** replace the 1% dividend estimate with the NIFTY 500 total-return series.
- **Start days:** report each strategy across all start days instead of a single run.
- **Data quality:** Yahoo has 24 one-day moves of 40% or more, and several look like splits or
  demergers it hasn't adjusted for. Removing them only takes momentum from 39.7% to 37.9%, so
  this isn't the main problem, but it does distort risk figures.
- **Better costs:** a slippage model that depends on liquidity (small caps cost more to trade).
- **India-specific data:** delivery % (NSE bhavcopy), FII/DII flows, and earnings-announcement
  dates.
- **Model:** a learning-to-rank objective (LightGBM LambdaRank), a turnover penalty in
  training, and regime-aware switching between momentum and reversal.
- **Statistics:** bootstrap confidence intervals on CAGR and Sharpe; a correction for testing
  several strategies.
