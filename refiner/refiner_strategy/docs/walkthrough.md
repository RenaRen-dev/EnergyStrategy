# Refiner Strategy: A Plain-Language Walkthrough

This document explains every piece of the refiner strategy in plain English.
No jargon without definition, no equations without intuition. If you can
read a stock quote, you can follow this.

---

## 1. The Big Picture

We built a trading strategy that buys and sells shares of US oil refiner
companies. The strategy looks at the "crack spread" -- the profit margin
refiners earn when they turn crude oil into gasoline and heating oil -- and
bets that when that margin is expanding, refiner stocks will go up, and
when it is shrinking, they will go down.

We tested this over 8.1 years of out-of-sample data (April 2018 through
May 2026) using a rigorous walk-forward methodology that prevents us from
accidentally using future information.

The honest summary: the strategy works, but it does not beat simply holding
the S&P 500. Its best use is as a tactical overlay on top of a passive
portfolio.

---

## 2. Why Refiner Stocks

Oil refiners are unusual companies. Their revenue depends almost entirely
on one number: the crack spread. This is not a metaphor -- it is an
accounting identity.

A refiner buys crude oil (priced at the WTI benchmark, ticker CL) and
sells two products: gasoline (RBOB benchmark, ticker RB) and heating oil
(ticker HO). The standard "3:2:1" crack spread formula says: for every
3 barrels of crude purchased, a refiner produces 2 barrels of gasoline
and 1 barrel of heating oil.

In dollars per barrel:

    Crack Spread = (2 x RB x 42 + HO x 42) / 3 - CL

The factor of 42 converts from price-per-gallon (how futures are quoted)
to price-per-barrel (42 gallons per barrel).

When the crack spread is high, refiners are making money on every barrel
they process. When it is low or negative, they are losing money. This
relationship is so direct that refiner stock prices track the crack spread
more closely than they track the overall stock market.

This gives us an edge: if we can predict the direction of the crack spread
one day ahead, we can predict which way refiner stocks will move.

---

## 3. The 7 Stocks We Trade

We trade a basket of 7 US-listed refiner stocks, weighted by market
significance:

| Ticker | Company | Weight | Why |
|--------|---------|--------|-----|
| VLO | Valero Energy | 25% | Largest independent refiner |
| MPC | Marathon Petroleum | 25% | Largest US refiner by capacity |
| PSX | Phillips 66 | 25% | Diversified downstream |
| DINO | HF Sinclair | 10% | Mid-cap, pure-play refiner |
| PBF | PBF Energy | 5% | East Coast refiner |
| DK | Delek US Holdings | 5% | Small-cap refiner |
| CVI | CVR Energy | 5% | Small-cap, high-yield refiner |

The top three (VLO, MPC, PSX) get 75% of the capital because they are
the most liquid and have the tightest bid-ask spreads. The smaller names
get less weight because they are harder and more expensive to trade.

All price data comes from yfinance at runtime. There are no bundled CSV
files -- the strategy downloads fresh data every time it runs.

---

## 4. Removing Market Noise (Beta-Hedging)

Refiner stocks move for two reasons: (1) changes in the crack spread
(the signal we want), and (2) changes in the overall stock market (noise
we want to remove).

If the S&P 500 drops 3% on a bad jobs report, refiner stocks will drop
too -- not because the crack spread changed, but because everything dropped.
We need to separate these two effects.

We do this with beta-hedging. Beta measures how much a stock moves with
the market. If VLO has a beta of 1.2, it means VLO typically moves 1.2%
for every 1% the S&P 500 moves.

The hedged return is:

    Hedged Return = Stock Return - Beta x SPY Return

This strips out the market component, leaving only the refiner-specific
signal.

**H5 Bug Fix:** An earlier version of the code computed beta using today's
return, which is circular -- you cannot use today's data to compute today's
hedge ratio. The fix is to lag the beta by one day using `.shift(1)`. This
means today's hedge ratio is computed from data available yesterday, which
is the correct causal ordering.

We compute beta as a rolling 60-day OLS regression slope, then shift it
by one day before applying it. This applies to both the basket-level beta
and every individual stock's beta.

---

## 5. Two Ways to Predict

We use two completely different methods to predict the direction of
refiner stocks. Each has strengths the other lacks.

### The DET Signal (Deterministic Rule)

This is a simple, time-tested trading rule used by physical commodity
trading desks:

1. Compute the 3:2:1 crack spread each day
2. Compute a 10-day simple moving average (SMA) of the crack spread
3. If the crack spread is above the SMA, the signal is "long" (+1)
4. If the crack spread is below the SMA, the signal is "short" (-1)
5. But only if the signal has been the same for 2 consecutive days
   (the persistence filter prevents whipsaws)

The persistence filter is important: without it, the signal would flip
back and forth every time the crack spread crossed the SMA, generating
excessive transaction costs. By requiring 2 days of confirmation, we
filter out noise crossovers.

The DET signal is applied with a one-day lag: today's trading decision
is based on yesterday's signal. This prevents look-ahead bias.

### The Chronos Signal (AI Forecast)

Chronos-2 is a foundation model for time-series forecasting, built by
Amazon. Think of it as a language model, but instead of predicting the
next word, it predicts the next number in a sequence.

We feed it two inputs:
- The history of hedged returns for each stock
- The history of the crack-spread Z-score (a normalized version of
  the crack spread)

It produces probabilistic forecasts: instead of saying "tomorrow's return
will be +0.3%", it gives us a range of quantiles (the 10th percentile,
20th, ..., 90th). This tells us both the expected direction AND how
confident the model is.

We fine-tune Chronos-2 using LoRA (Low-Rank Adaptation) adapters in a
walk-forward loop. Each fold uses 24 months of training data, and we
retrain every 6 months. This lets the model adapt to changing market
conditions without overfitting to any single period.

---

## 6. How Big a Position to Take

This is where the strategy gets interesting. We built six different
sizing schemes, each representing a different philosophy about how to
convert a forecast into a dollar position.

### The OLD Sizer (Probability Only)

The original, buggy version. It only looks at P(up) -- the probability
that tomorrow's return is positive -- and ignores everything else.

- If P(up) is between 0.40 and 0.60, do nothing (the "deadband")
- Otherwise, size proportional to how far P(up) is from 0.50

This is bad because it ignores how much the model expects the stock to
move (the magnitude) and how uncertain the forecast is (the spread
between quantiles).

### The NEW Sizer (Vol-Targeted)

The improved version with two gates:

**Gate 1 (Vol Floor):** If the forecast spread (q90 - q10) is too narrow,
the model is not making a meaningful prediction. Skip.

**Gate 2 (Consensus):** If the median forecast (q50) and P(up) disagree
on direction, the model is confused. Skip.

If both gates pass, size the position to target a specific daily
volatility (1% of capital).

### Worked Example: 5 Days of OLD vs NEW

Imagine VLO with a $25 allocation (25% of $100 notional). Here is what
happens over 5 days with each sizer:

**Day 1:** q10 = -0.5%, q50 = +0.30%, q90 = +1.0%, P(up) = 0.78

- OLD: P(up) = 0.78 is outside the deadband (0.40-0.60).
  Conviction = min(1.0, |0.78 - 0.5| x 5.0) = min(1.0, 1.4) = 1.0.
  Position = +1.0 x 1.0 x $25 = **+$25.00** (max long).
- NEW: Forecast vol = (1.0% - (-0.5%)) / 2.5631 = 0.586%.
  That exceeds the 0.5% vol floor, so Gate 1 passes.
  q50 > 0 and P(up) > 0.5, so Gate 2 passes.
  Edge = 0.30% / 0.586% = 0.512.
  Raw size = 0.512 / 1.0% = 51.2, clipped to 1.0.
  Position = +1.0 x $25 = **+$25.00**.

Both agree: strong long. But watch what happens when the forecast weakens.

**Day 2:** q10 = -1.5%, q50 = +0.05%, q90 = +1.6%, P(up) = 0.55

- OLD: P(up) = 0.55 is inside the deadband (0.40-0.60).
  Position = **$0.00** (flat).
- NEW: Forecast vol = (1.6% - (-1.5%)) / 2.5631 = 1.210%.
  Gate 1 passes. q50 > 0 and P(up) > 0.5: Gate 2 passes.
  Edge = 0.05% / 1.210% = 0.041.
  Raw size = 0.041 / 1.0% = 4.13, clipped to 1.0.
  Position = +1.0 x $25 = **+$25.00**.

OLD goes flat because P(up) is close to 0.50. NEW stays fully long because
even a tiny positive q50, when the forecast vol is high, produces a
meaningful edge ratio.

**Day 3:** q10 = -2.5%, q50 = -0.05%, q90 = +2.4%, P(up) = 0.45

- OLD: P(up) = 0.45 is inside the deadband.
  Position = **$0.00**.
- NEW: Forecast vol = (2.4% - (-2.5%)) / 2.5631 = 1.912%.
  Gate 1 passes. q50 < 0 and P(up) < 0.5: Gate 2 passes (both bearish).
  Edge = -0.05% / 1.912% = -0.026.
  Raw size = -0.026 / 1.0% = -2.61, clipped to -1.0.
  Position = -1.0 x $25 = **-$25.00** (max short).

Now they diverge completely. OLD is flat; NEW is max short.

**Day 4:** q10 = -0.8%, q50 = -0.50%, q90 = +0.3%, P(up) = 0.25

- OLD: P(up) = 0.25 is below 0.40, outside the deadband.
  Conviction = min(1.0, |0.25 - 0.5| x 5.0) = min(1.0, 1.25) = 1.0.
  Direction = -1.0 (p_up < 0.5).
  Position = -1.0 x 1.0 x $25 = **-$25.00** (max short).
- NEW: Forecast vol = (0.3% - (-0.8%)) / 2.5631 = 0.429%.
  That is below the 0.5% vol floor. Gate 1 blocks.
  Position = **$0.00**.

Reversed! OLD is max short while NEW is flat, because NEW recognises
that the narrow forecast spread means low confidence.

**Day 5:** q10 = -3.0%, q50 = +0.10%, q90 = +0.4%, P(up) = 0.40

- OLD: P(up) = 0.40 is on the deadband boundary (0.40 < 0.40 is false,
  so this is outside). Conviction = min(1.0, |0.40 - 0.5| x 5.0) = 0.50.
  Direction = -1.0. Position = -1.0 x 0.50 x $25 = **-$12.50**.
- NEW: Forecast vol = (0.4% - (-3.0%)) / 2.5631 = 1.326%.
  Gate 1 passes. But q50 = +0.10% > 0, while P(up) = 0.40 < 0.5.
  **Gate 2 blocks** (consensus disagreement).
  Position = **$0.00**.

This is the consensus gate in action. The median forecast says "up" but
the probability distribution says "down". When the two measures disagree,
the model is confused, and NEW wisely steps aside.

### NEW_CAP, DET, ENS_VETO, ENS_AVG

**NEW_CAP** adds a realised-volatility cap on top of NEW. If the stock's
trailing 20-day volatility exceeds 2%, positions are scaled down
proportionally. This prevents oversized bets during volatile periods.

**DET** ignores the AI forecast entirely and sizes purely on the
deterministic crack-spread signal. When DET says long, take a full
position; when DET says short, take a full short; when flat, hold nothing.
The vol cap also applies.

**ENS_VETO** requires both the AI (via NEW_CAP) and DET to agree. If
NEW_CAP says long and DET says long, take the position. If either
disagrees or is flat, do nothing. This is the most conservative scheme.

**ENS_AVG** uses Bates-Granger (1969) forecast combination: it averages
the Chronos q50 with the DET signal (scaled to a comparable magnitude).
This blends the two forecasts rather than vetoing on disagreement.

---

## 7. Combining the Two Signals

The DET signal and the Chronos forecast capture different information:

- DET is a trend-following rule. It works when crack spreads trend
  persistently in one direction.
- Chronos is a pattern-matching model. It can detect subtler relationships
  between the hedged return history and the Z-score.

The two ensemble methods combine them differently:

**ENS_VETO** is an AND gate: both must agree. This means it trades less
often but with higher conviction. It avoids the worst losses from either
signal acting alone.

**ENS_AVG** is a weighted average: it blends the Chronos median forecast
with the DET signal (the DET signal contributes 0.5 x DET_SIGNAL_MAG =
0.25 basis points to the combined q50). This smooths out noise from
either signal.

In practice, ENS_VETO produces the best Sharpe ratio (0.51) and the
shallowest drawdown (-37%), while ENS_AVG produces the highest annual
return (+8.97%) but with a deeper drawdown (-44%).

---

## 8. How We Test Honestly

Backtesting is dangerous. It is trivially easy to build a strategy that
looks brilliant in hindsight but fails in real trading. We took specific
steps to prevent this.

### Purged Walk-Forward Validation

Following Lopez de Prado (2018, Chapter 7), we use purged walk-forward
optimisation:

1. Divide the out-of-sample period (2018-04 to 2026-05) into 17
   non-overlapping 6-month test windows
2. For each test window, train the AI model on the preceding 24 months
3. Insert a 5-day purge gap between the training and test periods

The purge gap prevents information leakage. Without it, the last few
training days and first few test days share overlapping market conditions,
which can inflate apparent performance.

### Strict Temporal Ordering

Every prediction uses `history = df[df.index < T]` -- strict less-than,
never less-than-or-equal. This means today's prediction cannot use any
information from today. Combined with the one-day lag on the DET signal
and the one-day lag on beta, the strategy makes decisions using only
information that was available at the previous close.

### Deterministic Seeds

Each fold uses a deterministic random seed: `fold_seed = LORA_BASE_SEED +
fold_idx`. This means results are exactly reproducible: run the same code
twice, get the same numbers. Without per-fold seeds, stochastic variation
in the fine-tuning could make results look better or worse on any given
run.

---

## 9. The Results

Here is the full results table, covering 8.1 years of truly out-of-sample
data (2044 trading days, 17 walk-forward folds, 10 basis points round-trip
transaction cost):

| Scheme | Ann Return | Sharpe | Max Drawdown |
|--------|-----------|--------|--------------|
| OLD | -4.15% | -0.29 | -- |
| NEW | -1.97% | +0.14 | -- |
| NEW_CAP | +2.87% | +0.24 | -- |
| DET | +8.87% | +0.45 | -60% |
| ENS_VETO | +7.78% | +0.51 | -37% |
| ENS_AVG | +8.97% | +0.50 | -44% |
| SPY (benchmark) | +13.94% | +0.82 | -34% |

Key observations:

- The progression from OLD to ENS_AVG shows iterative improvement: each
  fix and enhancement adds real value.
- DET alone produces +8.87% with a 0.45 Sharpe, but its -60% drawdown
  is unacceptable for most allocators.
- ENS_VETO cuts that drawdown nearly in half (-37%) while only giving up
  1pp of annual return.
- None of the standalone schemes beat SPY's +13.94% / 0.82 Sharpe.

---

## 10. What This Proves

Three lessons emerge:

**Lesson 1: The crack spread signal is real.** DET alone produces a 0.45
Sharpe over 8.1 years. This is not a fluke -- it reflects a genuine
economic relationship between refining margins and refiner equity prices.

**Lesson 2: The AI adds value as a filter, not a standalone signal.**
Chronos alone (the OLD and NEW schemes) underperforms. But when combined
with DET via ENS_VETO, it reduces drawdowns by 23 percentage points
while maintaining most of the return. The AI's contribution is knowing
when NOT to trade.

**Lesson 3: Position sizing matters more than signal generation.** The
same DET signal, sized six different ways, produces Sharpes ranging from
-0.29 to +0.51. The sizing scheme is not an afterthought -- it is half
the strategy.

---

## 11. The Four Bugs We Found and Fixed

During development, four significant bugs were identified and corrected.
Each is documented because understanding the bugs is as instructive as
understanding the strategy.

### H1: Z-Score Computed Per-Slice

**The bug:** The original code computed a 256-day rolling Z-score
separately within each walk-forward fold's data slice. This created a
discontinuity at slice boundaries: the Z-score would jump when a new
fold started because the rolling window restarted.

**The fix:** Compute a single unified Z-score across the entire crack
spread history, then slice the dates for each fold. The Z-score is always
computed from the same continuous series.

### H4: Hit Rate Used Target Size

**The bug:** The hit rate calculation used `target_size` (the position
decided at T-1 close, which will be held during T+1) instead of
`effective_size` (the position actually held during T). This overstated
accuracy by approximately 3 percentage points because it attributed
today's return to tomorrow's intended position.

**The fix:** Track both `target_size` and `effective_size` in the trade
log. The hit rate uses `effective_size`: was the position that was
actually held during day T on the right side of that day's move?

### H5: Beta Used Today's Return

**The bug:** The rolling beta was computed using today's return in the
regression, then immediately used to hedge today's return. This is
circular -- you cannot use information from today to remove today's
market effect.

**The fix:** Apply `.shift(1)` to the rolling beta before using it.
Today's hedge ratio uses beta computed from data up through yesterday.

### H3: Same Seed for All Folds

**The bug:** All 17 fine-tuning folds used the same random seed (42).
This meant the LoRA fine-tuning was not truly independent across folds:
the same initialisation could produce correlated errors.

**The fix:** Use `fold_seed = LORA_BASE_SEED + fold_idx`, so each fold
gets a unique but deterministic seed. Results are still reproducible,
but folds are independent.

---

## 12. What Else We Tried (Alpha Search Ledger)

Before arriving at the current strategy, we tested 9 additional alpha
sources. All were rejected. This section documents them because negative
results are as important as positive ones -- they constrain the space of
what works.

| # | Strategy | Reference | Sharpe |
|---|----------|-----------|--------|
| 1 | Cross-sectional momentum (CSMOM) | Jegadeesh-Titman 1993 | -0.99 |
| 2 | Margin-leveraged overlay | -- | +0.26pp marginal |
| 3 | Calendar mean-revert (M1-M3) | Carter-Rausser-Schmitz 1983 | -0.30 |
| 4 | Calendar momentum (flipped) | -- | +0.19 |
| 5 | Long-only AI defense | -- | -8pp vs SPY |
| 6 | Oil-shock momentum | Driesprong 2008 | -0.06 |
| 7 | Oil-shock fade | -- | -0.48 |
| 8 | Crack divergence (RB vs HO) | Pirrong 2012 | -0.03 |
| 9 | Basket-vs-XLE pairs | Engle-Granger 1987 / Avellaneda-Lee 2010 | -0.46 |

### The Story of Attempt #9 (Basket-vs-XLE Pairs Trading)

This one deserves a detailed post-mortem because it went through three
iterations before we killed it.

**First implementation (Sharpe: -0.68).** The idea was to trade the
spread between our 7-stock basket and XLE (the Energy Select Sector SPDR
ETF). When the spread is wide, short the basket and go long XLE; when
narrow, do the opposite.

The first attempt had three implementation bugs:
- Used a unit beta (1.0) instead of estimating the hedge ratio
- Traded only the basket leg, not the spread
- Used the wrong sign on the hedge

**Second implementation (Sharpe: -0.46).** After fixing the bugs, we:
- Estimated a rolling 252-day beta between the basket and XLE
- Traded the actual spread (long basket / short XLE or vice versa)
- Computed proper hedged returns

The Sharpe improved from -0.68 to -0.46, but was still deeply negative.

**Statistical kill shot.** We ran an Engle-Granger cointegration test
on the basket-vs-XLE spread. The test returned a p-value of 0.27 --
far above the 0.05 threshold needed to reject the null hypothesis of
no cointegration. Furthermore, the estimated half-life of mean reversion
was 210 trading days (nearly a year), meaning even if the spread does
mean-revert, it does so too slowly to trade profitably at daily frequency.

Conclusion: the basket and XLE are correlated but not cointegrated.
Pairs trading only works when the spread is stationary, and this one
is not.

---

## 13. Does This Beat SPY?

No, not as a standalone strategy.

The best standalone scheme (ENS_AVG) returns +8.97% annually with a
0.50 Sharpe. SPY returns +13.94% with a 0.82 Sharpe. We trail by
5 percentage points on return and 0.32 on Sharpe.

The more interesting question is: **can the refiner strategy add value
on top of a passive SPY portfolio?** This is the SPY-default overlay.

When the strategy has a position, some capital is in refiners. When it
does not, that capital sits in SPY earning the market return. The
combined portfolio looks like:

| Scheme | BPS RT | Ann Return | Sharpe | Edge vs SPY |
|--------|--------|-----------|--------|-------------|
| ENS_VETO | 10 | +16.37% | +0.74 | +2.36pp |
| ENS_AVG | 10 | +15.89% | +0.68 | +1.88pp |
| DET | 10 | +15.21% | +0.61 | +1.20pp |
| ENS_VETO | 15 | +15.98% | +0.72 | +1.97pp |
| ENS_VETO | 20 | +15.59% | +0.70 | +1.58pp |
| ENS_VETO | 25 | +15.20% | +0.67 | +1.19pp |

Only ENS_VETO at 10 bps round-trip convincingly beats SPY by +2.36
percentage points annually. At higher transaction costs, the edge
shrinks. At 25 bps, ENS_VETO still adds +1.19pp, but the margin is
thin enough to be within estimation error.

The honest answer: this strategy is a modest source of alpha that
works best as a satellite allocation alongside a passive core.

---

## 14. Honest Limitations

**#0: Does not beat SPY.** Restating for emphasis. If you can only hold
one thing, hold SPY. This strategy is a complement, not a replacement.

**#1: Roll noise in futures.** The crack spread is computed from
front-month continuous contracts (CL=F, RB=F, HO=F). These have roll
gaps where the contract switches from one expiry to the next. The crack
spread inherits this noise. We do not apply any back-adjustment because
the DET signal is a relative measure (crack vs SMA) that partially
cancels the discontinuity, but the Z-score does not.

**#2: Optimistic transaction costs.** We assume 10 basis points
round-trip for all stocks. For VLO, MPC, and PSX, this is reasonable.
For CVI and DK (small-cap, lower liquidity), real costs could be
2-5x higher. The strategy allocates only 5% to each of these names,
but the cost assumption is still aggressive.

**#3: No slippage model.** We assume positions are filled at the closing
price. In reality, large orders move the market. For a retail-sized
account this is fine; for institutional size, market impact would erode
returns.

**#4: Survivorship bias.** The 7-stock universe was selected with the
benefit of hindsight. Companies that went bankrupt, were acquired, or
delisted during the backtest period are not included. This biases results
upward.

**#5: Single-sector concentration.** The strategy is 100% energy sector.
Sector-wide drawdowns (COVID crash in March 2020, oil price war) hit all
7 stocks simultaneously. The -60% max drawdown on DET reflects this.

**#6: Chronos model risk.** The AI component depends on a specific
pre-trained model (amazon/chronos-2). If the model is updated, deprecated,
or behaves differently on new data distributions, the fine-tuned adapters
may stop working. There is no guarantee that future versions of Chronos
will be compatible with our LoRA weights.

**#7: No live trading validation.** Everything in this report is from
backtests. Paper trading and live execution introduce additional failure
modes: API outages, data delays, order routing issues, margin calls.
None of these are captured in the backtest.

---

## 15. The Bottom Line

The refiner strategy demonstrates that the crack spread is a genuine,
tradeable signal for refiner equity returns. Over 8.1 years of
out-of-sample testing, a deterministic SMA crossover rule combined with
an AI-based defensive filter (ENS_VETO) produces a 0.51 Sharpe ratio
with a -37% maximum drawdown. As a standalone strategy it does not beat
SPY, but as a tactical overlay on a passive SPY portfolio, it adds
+2.36 percentage points of annual return at optimistic transaction cost
assumptions. The main value of the work is not the strategy itself but
the methodology: rigorous walk-forward testing, honest documentation of
bugs and failed attempts, and a clear-eyed assessment of limitations.
