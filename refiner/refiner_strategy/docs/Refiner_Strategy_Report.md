# AI Refiner Strategy: Full Project Report
## Target: 7 US Oil Refiner Equities | Horizon: 1 Business Day
**Author:** Tianyu Shi | **Date:** May 2026

---

# Part 1: Strategy Overview

*Goal: Build a long/short equity strategy on US oil refiners that strips out broad market noise and exploits the refining margin cycle.*

---

## 1.1 — The Core Idea

Oil refiners (VLO, MPC, PSX, DINO, PBF, DK, CVI) make money on the **crack spread** — the difference between what they buy (crude oil) and what they sell (gasoline, heating oil). When the crack spread is wide, refiners are profitable. When it collapses, their margins disappear.

This strategy trades on that cycle in two layers:

1. **Deterministic layer (DET):** A rule-based signal based on crack spread momentum — the same signal used by the Houston Products Desk. Fast, transparent, always running.
2. **AI layer (Chronos-2):** An AI time-series model that forecasts next-day hedged returns using the crack spread as a covariate. Adds probabilistic conviction on top of the DET rule.

Both signals feed into **6 sizing schemes** tested in a full A/B backtest. The goal is to find which combination of rule + AI produces the best risk-adjusted return.

---

## 1.2 — Universe & Capital Weights

| Ticker | Company | Weight |
|--------|---------|--------|
| VLO | Valero Energy | 25% |
| MPC | Marathon Petroleum | 25% |
| PSX | Phillips 66 | 25% |
| DINO | HF Sinclair | 10% |
| PBF | PBF Energy | 5% |
| DK | Delek Group | 5% |
| CVI | CVR Energy | 5% |

**Why these weights?** VLO, MPC, and PSX dominate — they are the three largest independent US refiners by throughput capacity and have the deepest liquidity. The smaller names (DINO, PBF, DK, CVI) receive smaller allocations because they are more idiosyncratic and harder to execute.

---

# Part 2: Data Infrastructure

*Goal: Build a clean dataset where every column is economically meaningful and free of look-ahead bias.*

---

## 2.1 — Futures Data & Crack Spread (`data/futures_loader.py`)

**What:** Downloads front-month WTI Crude (CL), RBOB Gasoline (RB), and Heating Oil (HO) futures from yfinance. Computes the **3:2:1 crack spread**.

**The 3:2:1 formula:**

```
Crack Spread ($/bbl) = (2 × RB_price × 42  +  HO_price × 42) / 3  −  CL_price
```

The 3:2:1 ratio means: for every 3 barrels of crude input, a refinery produces approximately 2 barrels of gasoline and 1 barrel of heating oil.

**Multiply by 42:** RB and HO futures are quoted in dollars per gallon. Multiply by 42 (gallons per barrel) to convert them to the same $/bbl unit as crude.

**Why the crack spread matters:**  
The crack spread is the *real-time profit margin* of refining. It tells you whether the market expects refiners to be profitable right now. A rising crack = improving margins = buy refiners. A falling crack = margin compression = short refiners.

**yfinance column ordering gotcha:**
```python
# We ask for ["CL=F", "RB=F", "HO=F"]
# But yfinance returns columns alphabetically: [CL=F, HO=F, RB=F]
close.columns = ["CL", "HO", "RB"]   # must match the alphabetical order
```
Getting this wrong would silently swap gasoline and heating oil prices, corrupting every downstream calculation.

---

## 2.2 — Equity Prices & Returns (`data/build_dataset.py`)

**What:** Downloads adjusted-close prices for all 7 refiners plus SPY (the S&P 500 ETF) from yfinance. Computes daily log returns.

**Why SPY?** Refiners are not pure commodity plays — they carry significant market beta. When the S&P sells off, refiners fall too, regardless of the crack spread. To isolate the refining-margin signal we need to strip out this market exposure. SPY is our market proxy.

---

## 2.3 — Hedged Returns: Stripping Out Market Beta

**The problem:** If VLO has a beta of 1.2, then 1.2% of every 1% SPY move is just "market noise" — it has nothing to do with refiner margins. Trading on raw VLO returns means you are mostly just trading the S&P 500.

**The fix — beta-hedged returns:**

```
Hedged_Return(t) = Raw_Return(t)  −  Beta(t−1) × SPY_Return(t)
```

Where Beta is a 60-day rolling OLS estimate:

```
Beta(t) = Cov(Asset_Returns, SPY_Returns) / Var(SPY_Returns)
          computed over the last 60 trading days
```

**H5 correctness fix — beta is lagged:**  
We use Beta(t−1), not Beta(t). Computing beta with today's return and then using it to hedge today's return would be look-ahead bias. Today's beta must use only data available at yesterday's close.

**End-to-end example:**

```
VLO Raw Return today:    +2.1%
SPY Return today:        +1.5%
VLO Rolling Beta (t−1):  1.20

VLO Hedged Return = +2.1%  −  (1.20 × 1.5%)  =  +2.1%  −  1.8%  =  +0.3%
```

The hedged return tells us: VLO earned +0.3% *above and beyond* what the market predicted. That is the refiner alpha we want to trade.

---

## 2.4 — Crack Spread Z-Score

**What:** A 256-day rolling Z-score of the crack spread.

```
Crack_Z_Score(t) = (Crack_Spread(t) − Rolling_Mean_256d) / Rolling_Std_256d
```

Clipped to [-3.0, +3.0].

**H1 correctness fix — unified Z-score:**  
The Z-score is computed once on the *full* crack series from 2014, not recomputed for each fold or test window. Re-normalizing per fold would reset the baseline, making a +$5/bbl crack spread "look normal" in one fold and "look high" in another. A unified Z-score preserves the absolute level of the margin cycle.

**What the Chronos model sees:**  
Chronos-2 receives two inputs: `[hedged_return × 100, crack_Z_score]`. The Z-score tells the model where we are in the refining margin cycle. A high Z-score (+2.0) says "margins are unusually strong right now" — providing context the model uses to sharpen its return forecast.

---

## 2.5 — Data Pipeline: End-to-End Example

Follow one trading day through every step:

```
╔══════════════════════════════════════════════════════════════════════╗
║  INPUT:  Date = 2023-06-15 (morning, before market open)           ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
┌─────────────────────────────┬────────────────────────────────────────┐
│  STEP 2.1: Crack Spread     │  WHY: Raw VLO price means nothing     │
│                             │  without knowing refiner margins.     │
│  CL = $70.20/bbl            │                                        │
│  RB = $2.65/gal × 42 = $111 │  Crack Spread = (2×$111 + 1×$88) / 3 │
│  HO = $2.10/gal × 42 = $88  │                = $103.33 - $70.20     │
│  Crack Spread = $33.13/bbl  │                = $33.13/bbl           │
└─────────────────────────────┴────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────┬────────────────────────────────────────┐
│  STEP 2.4: Z-Score          │  WHY: Is $33/bbl high or low?         │
│                             │  We need context. Z-score tells us    │
│  Mean_256d = $27.50         │  relative to recent history.          │
│  Std_256d  = $5.20          │                                        │
│  Z = (33.13 - 27.50) / 5.20 │  Z = +1.08 → margins are 1 std dev   │
│    = +1.08                  │  ABOVE their 1-year average.          │
└─────────────────────────────┴────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────┬────────────────────────────────────────┐
│  STEP 2.3: Hedged Return    │  WHY: VLO went up today, but so did   │
│                             │  the whole market. We strip out the   │
│  VLO Raw Return:   +1.8%   │  market move to isolate the refiner   │
│  SPY Return:       +1.2%   │  alpha.                                │
│  VLO Beta (t−1):   1.15    │                                        │
│                             │  Hedged = +1.8% - (1.15 × 1.2%)      │
│  VLO Hedged Ret:   +0.42%  │           = +0.42% of pure alpha      │
└─────────────────────────────┴────────────────────────────────────────┘
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ML-READY:  Crack_Z = +1.08  |  VLO_Hedged_Return = +0.42%        ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

# Part 3: The Deterministic Signal (DET)

*Goal: A fast, interpretable signal that any trader can audit — no black box.*

---

## 3.1 — How It Works (`signals/det_signal.py`)

The DET signal replicates the rule-based approach used by the Houston Products Desk. It has four steps:

**Step 1 — 3:2:1 Crack Spread**  
Same formula as Part 2: how much a refiner earns per barrel today.

**Step 2 — 10-day Simple Moving Average**  
Smooth the noisy daily crack spread into a trend.

**Step 3 — Crossover Signal**
```
raw_sig = +1  if  Crack > SMA10   (margins trending up   → bullish)
raw_sig = −1  if  Crack < SMA10   (margins trending down → bearish)
raw_sig =  0  if  Crack = SMA10
```

**Step 4 — 2-day Persistence Filter**  
The raw signal fires on the first crossover, even if it's a one-day blip. The persistence filter requires the crossover to **hold for 2 consecutive days** before the signal flips:

```
Day 1: Crack > SMA → raw_sig = +1, det_sig = 0   (not confirmed yet)
Day 2: Crack > SMA → raw_sig = +1, det_sig = +1  (confirmed → GO LONG)
Day 3: Crack < SMA → raw_sig = −1, det_sig = 0   (reversal not confirmed)
Day 4: Crack < SMA → raw_sig = −1, det_sig = −1  (confirmed → GO SHORT)
```

**Why the persistence filter?**  
Crack spreads are volatile day to day — they can cross their SMA on one day and revert the next. Trading every crossover creates excessive churn and transaction costs. The 2-day filter prevents whipsaws by requiring conviction before acting.

**Lag rule:**  
The DET signal is always lagged by 1 day before trading:
```python
det_sig.shift(1)  # today's trade uses yesterday's signal
```
This is because the signal is built from yesterday's closing futures prices. Using it before the market opens is correct; using it same-day would be look-ahead bias.

---

## 3.2 — DET Signal Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SMA_WINDOW` | 10 days | Rolling average window |
| `SMA_MIN_PERIODS` | 5 | Minimum days to compute SMA |
| `CONFIRM_DAYS` | 2 | Days crossover must hold before signal flips |

---

# Part 4: Chronos-2 AI Signal

*Goal: Use a pre-trained AI model to generate a probabilistic forecast of next-day hedged return — providing conviction the DET rule cannot.*

---

## 4.1 — What Chronos-2 Is

**Model:** `amazon/chronos-2` (120M parameters)  
**Pre-trained on:** Billions of time-series observations across energy, finance, weather, and retail  
**Role:** The AI forecaster — sees the recent history of hedged returns and crack spread, predicts the next day's return distribution.

Chronos-2 works like **ChatGPT, but for numbers instead of text**.

ChatGPT reads *"The weather today is sunny and warm, so tomorrow will be..."* and predicts *"sunny."* Chronos-2 reads a sequence of daily returns and crack Z-scores, then predicts what tomorrow's return is likely to be:

```
History:   [+0.3%, +0.1%, −0.2%, +0.5%, −0.1%]  ← hedged returns
Covariates:[+0.8,  +0.9,  +1.1,  +1.0,  +1.2]   ← crack Z-scores
                                                         ↓
Model predicts: "Tomorrow's return will be +0.2%–+0.4% with 78% probability positive"
```

---

## 4.2 — What the Model Outputs

Unlike the DET signal which just says "long" or "short," Chronos-2 outputs a **full probability distribution** over tomorrow's return:

```
Quantile forecasts for next-day VLO hedged return:
  q10 = −0.48%   (pessimistic — 10% chance return is worse than this)
  q20 = −0.21%
  q30 = −0.08%
  q40 = +0.03%
  q50 = +0.12%   (median — most likely outcome)
  q60 = +0.22%
  q70 = +0.31%
  q80 = +0.44%
  q90 = +0.61%   (optimistic — 10% chance return is better than this)

  p_up = 0.67    (67% of quantiles are above 0 → 67% chance of positive return)
```

**q50 vs p_up — two different questions:**
- `q50` answers *"where is the centre of the return distribution?"* (level and direction)
- `p_up` answers *"what is the probability the return is positive?"* (conviction)

They are related but not the same. A distribution skewed left can have `q50 > 0` but `p_up` only 0.55 — the median is barely positive but the distribution has a heavy left tail. The sizing schemes use both to make better decisions than either alone.

---

## 4.3 — Input Format

The model receives a **2-variate context window**:
- **Variate 0 (target):** `hedged_return × 100` — the return series the model is forecasting
- **Variate 1 (covariate):** `Crack_Z_Score` — provides the macro refining margin context

```
Input shape: (1, 2, T)
             │  │  └── timesteps (up to 512 days of history)
             │  └───── 2 variates (return + crack Z)
             └──────── batch dimension (1 series)
```

The crack Z-score is not being predicted — it is a **conditioning variable** that helps the model understand the current market regime. A model that sees "crack Z = +2.0 and rising" learns to forecast differently than one that sees "crack Z = −1.5 and falling."

---

## 4.4 — Walk-Forward LoRA Fine-Tuning (`finetune/walkforward.py`)

**What:** Rather than using Chronos-2 zero-shot, we fine-tune it with **LoRA adapters** on historical refiner data. This adapts the general-purpose model to the specific dynamics of refiner stocks.

**LoRA (Low-Rank Adaptation):** Instead of retraining 120M parameters (slow, expensive, risks overfitting), LoRA adds a small set of adapter weights (~0.1% of the model) that shift the model's behavior for this domain. It is like giving an expert generalist a short brief on refiner stocks — they don't forget their general knowledge, they just add domain-specific context.

### The 17-Fold Walk-Forward Design

The key challenge: we cannot train on future data. We use a **purged walk-forward** design:

```
Fold 0:
  ─────────────────────────────────────────────────────────────────
  │← 24 months train →│ purge │←── 6 months test ──→│
  2016-04    →    2018-03    gap   2018-04  →  2018-09
  
Fold 1:
                              │← 24 months train →│ purge │← 6 months test →│
                              2016-10    →    2018-09    gap  2018-10 → 2019-03
                              
...  (17 folds total, covering 2018-04 through ~2026)
```

**Why 3 design rules?**

| Rule | Value | Why |
|------|-------|-----|
| Train window | 24 months | Enough data to capture a full refining margin cycle |
| Test window | 6 months | Long enough for statistical significance, short enough to stay relevant |
| Purge gap | 5 days | Prevents data leakage: returns are correlated day-to-day, so nearby dates could "contaminate" the test set if included in training |

**Per-fold seeds (H3 fix):**  
Each fold uses a deterministic seed: `fold_seed = 42 + fold_idx`. This ensures results are reproducible — re-running fold 3 always produces identical weights, regardless of which other folds have run.

---

# Part 5: The 6 Sizing Schemes

*The core problem all sizers solve: given a Chronos-2 forecast and a crack-spread signal, how much capital should we commit to a refiner stock today?*

---

## The Central Finding

Before examining individual schemes, the most important result must be stated upfront:

**All six schemes produce directional hit rates between 50.9% and 53.0% — essentially indistinguishable. The Sharpe ratio improves 7× from 0.07 (OLD) to 0.51 (ENS_VETO). Every basis point of that improvement comes from sizing discipline, not from predicting direction more accurately.**

The progression OLD → ENS_AVG is not a story of better AI. It is a story of better risk management. Hit rates barely move. Sharpe ratios do not. This is the central lesson of the backtest.

---

## 5.1 — OLD: Probability-Only Sizer (Naïve baseline)

```
if 0.40 < p_up < 0.60:  return 0.0   # deadband
conviction = |p_up - 0.5| × 5
size = direction × conviction × capital
```

**Economic logic:** Pure binary bet. If the model says >60% chance of going up, go long; <40%, go short. Conviction scales linearly with probability edge.

**Why it fails:**
- Ignores how volatile the asset is. A 70% conviction bet in a 5%-daily-vol stock is far riskier than the same bet in a 0.5%-vol stock.
- No risk budget → position sizes are arbitrary relative to the portfolio's risk tolerance.
- Equivalent to Kelly betting with no variance adjustment, which leads to ruin over time.

---

## 5.2 — NEW: Vol-Targeted Sizer (Moskowitz-Ooi-Pedersen 2012)

```
forecast_vol = (q90 - q10) / 2.56     # implied σ from quantile spread
if q50 and p_up disagree in direction: return 0.0   # consensus gate
edge  = q50 / forecast_vol             # Sharpe-like ratio
size  = edge / TARGET_DAILY_VOL × capital
```

**Economic logic:** Targets a constant daily portfolio volatility (1% per day, set in config). Inspired by time-series momentum literature.

- `forecast_vol` = implied uncertainty of the Chronos prediction (wide quantile spread = uncertain forecast = smaller position)
- `edge / forecast_vol` = a forward-looking Information Ratio — expected return per unit of forecast risk
- Dividing by `TARGET_DAILY_VOL` translates that ratio into a position that contributes exactly 1% vol to the portfolio if the model is right

**Why this is better:** It answers the question "how confident is the model?" not just "which direction?". A tight quantile spread (model is sure) → bigger bet. Wide spread (model is uncertain) → smaller bet. Rational risk allocation.

**Consensus gate:** Rejects trades where the median forecast (q50) and probability (p_up) point in opposite directions — a self-contradiction in the model's output, discarding low-quality signals.

---

## 5.3 — NEW_CAP: Vol-Targeted + Realized-Vol Cap (Drawdown control)

```
base = size_new(...)
if realized_vol > VOL_CAP (2%/day):
    size = base × (VOL_CAP / realized_vol)
```

**Economic logic:** Adds a realized volatility brake on top of the forecast-vol sizing.

The NEW sizer uses predicted vol from Chronos quantiles. But markets can spike — a refiner stock might suddenly move 4%/day due to an oil shock. NEW_CAP de-levers automatically when the market itself is volatile, regardless of what Chronos thinks.

This is the risk management layer: even if the model is confident (tight quantiles), if the stock is whipping around 3× its normal range, you cut size. Similar to how volatility-targeting funds reduce exposure after the VIX spikes.

---

## 5.4 — DET: Pure Deterministic Signal (Rule-based trading desk)

```
if crack_spread > 10-day SMA:  size = +1 × capital × weight
if crack_spread < 10-day SMA:  size = -1 × capital × weight
# 2-day persistence filter prevents whipsaws
```

**Economic logic:** A direct implementation of the refinery crack spread trade — the physical economics of turning crude oil into gasoline/diesel.

- Crack spread wide (gasoline > crude): refining margins are high → refiners earn more → long refiners
- Crack spread narrow/negative: margins compressed → short refiners

This is fundamentals-based momentum: not ML, no quantiles, no probability. It encodes the actual commodity desk logic used by Houston energy traders. The 10-day SMA smooths noise; the 2-day confirmation filter prevents reacting to single-day blips.

**Why it outperforms everything (DET: Sharpe 0.44–0.98):** The crack spread is a direct economic driver of refiner profitability. The signal has genuine informational content that Chronos-2 must reconstruct indirectly from price history. The deterministic signal is the ground truth the model is trying to approximate.

---

## 5.5 — ENS_VETO: Ensemble with Veto (Disagreement = no trade)

```
base = size_new_cap(...)       # Chronos says X
if det_sig == 0: return 0.0    # DET flat → don't trade
if (base > 0) != (det_sig > 0): return 0.0   # disagree → veto
return base                     # both agree → Chronos-sized position
```

**Economic logic:** Require consensus between two independent forecasters before committing capital.

This is a classic signal confirmation approach from multi-factor investing:
- Chronos captures statistical patterns in price/crack history
- DET captures the physical commodity economics

If they agree: high-conviction trade → use Chronos vol-targeted sizing
If they disagree: uncertainty is high, sit out — the two information sources are contradicting each other

**Why this has the best Sharpe (0.51–0.68):** It dramatically reduces MaxDD (−27.7% vs −54.1% for DET alone) by avoiding trades where the fundamental signal and the ML signal conflict. You give up some return for a much smoother equity curve.

---

## 5.6 — ENS_AVG: Bates-Granger Forecast Combination (1969 forecast averaging)

```
avg_q50 = 0.5 × (chronos_q50 + det_sig × DET_SIGNAL_MAG)
edge    = avg_q50 / forecast_vol
size    = clipped(edge / TARGET_DAILY_VOL) × capital
```

**Economic logic:** Blend two forecasts into one signal rather than requiring them to agree.

From Bates & Granger (1969): "A combination of forecasts typically beats any single forecast." The blended median:
- Takes Chronos's q50 (ML quantile prediction)
- Averages it with `det_sig × 0.005` (DET rescaled to match the return magnitude)
- Uses the blend as the expected return in the vol-targeting formula



---

## 5.7 — Summary: The Economic Design Ladder

```
OLD       →  "what direction?"         No risk control
NEW       →  "how confident?"          Forecast-risk sizing
NEW_CAP   →  "is the market calm?"     Realized-risk brake
DET       →  "what do fundamentals say?" Physical commodity economics
ENS_VETO  →  "do both agree?"          Conservative consensus
ENS_AVG   →  "what's the blend?"       Diversified combination
```

---

## 5.8 — Transaction Costs

All schemes apply friction on **position changes**, not on the level of the position itself:

```
friction = |target_size − effective_size| × (10 bps / 10,000)
```

**10 bps = 0.10%** per unit of notional changed. A strategy that flips from +$25 to −$25 pays 10 bps on $50 of change ($0.05). Applied every day a position changes; no cost for holding a static position.

This models the bid-ask spread and commission cost of executing equity trades. The borrow cost for short positions is excluded here — see Script 05 (`spy_default_simulation`) for a full sweep of both transaction costs and borrow costs against the SPY baseline.

---

# Part 6: A/B Evaluation Framework

*Goal: Run all 6 schemes through an identical backtest to find which adds the most value.*

---

## 6.1 — The A/B Harness (`evaluation/ab_runner.py`)

All 6 schemes run through the **same** loop with the **same** accounting rules. No scheme gets an advantage from a different implementation.

**Correctness invariant — strict look-ahead prevention:**
```python
history = df[df.index < T]   # strict < : today's data excluded
```
On every test day T, the model only sees data from *before* T. Never same-day or future data.

**Two execution modes sharing identical accounting logic:**
- `run_ab_zero_shot`: Runs Chronos live on each day (no fine-tuning). Good for baseline comparison.
- `replay_with_predictions`: Reads predictions pre-saved from walk-forward folds. Used to evaluate the LoRA fine-tuned model.

---

## 6.2 — One Day in the Backtest Loop

```
For each trading day T in test_dates:

  1. history = all data strictly before T
  2. For each ticker:
       - Run Chronos on history (or load pre-saved prediction)
       - Get {q10, ..., q90, p_up}
  3. Compute realized vol from trailing 20-day hedged return std
  4. Get DET signal from lagged crack spread
  5. For each scheme:
       a. target_size = sizer(pred, det_sig, ...)    ← what we want to hold
       b. effective_size = what we held yesterday    ← what earns/loses today
       c. PnL = effective_size × actual_return − friction(|target − effective|)
       d. Record trade, update position to target_size
  6. Move to T+1
```

**H4 correctness fix — effective_size:**  
The position that earns today's return is what we held at yesterday's close (`effective_size`), not what we just decided (`target_size`). The decision made today only takes effect at today's close, earning tomorrow's return. Using `target_size` for today's PnL would be look-ahead bias in the execution model.

---

## 6.3 — Performance Metrics (`evaluation/metrics.py`)

All metrics are computed from the daily PnL series:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Ann. Return** | `mean_daily_ret × 252` | Annualized expected return |
| **Sharpe** | `mean_daily_ret / std × √252` | Return per unit of total risk |
| **Sortino** | `mean / downside_std × √252` | Return per unit of downside risk |
| **Max Drawdown** | `min((cum_ret − running_max) / running_max)` | Worst peak-to-trough decline |
| **Calmar** | `ann_ret / |max_drawdown|` | Return per unit of drawdown risk |
| **Hit Rate** | `% of active days where position × return > 0` | Directional accuracy |

**Hit rate uses `effective_size` (H4):**  
A trade "hits" when the position held at open earns a positive return by close. Using `target_size` (the decision) instead of `effective_size` (the holding) would misattribute gains to decisions that hadn't yet taken effect.

---

# Part 7: Running the Strategy

*How to execute the pipeline from data to results.*

---

## 7.1 — Execution Order

### Step 0 — Smoke tests (verify your environment)
```
cd refiner/refiner_strategy
python tests/smoke_chronos.py      # confirms Chronos-2 loads and predicts
python tests/smoke_fit.py          # confirms LoRA fine-tuning API works
python tests/test_sizing.py        # confirms all 6 sizing schemes are correct
```

### Step 1 — Build the master dataset
```
python scripts/01_build_datasets.py
```
Downloads futures + equity prices from yfinance. Computes crack spread, Z-score, rolling betas, and hedged returns. Saves `outputs/<run_dir>/datasets/master.csv`.

**Runtime:** ~1–2 minutes (network dependent).

### Step 2 — Walk-forward LoRA fine-tuning
```
python scripts/02_run_finetune_walkforward.py --run-dir outputs/<run_dir>
```
Runs 17 purged folds. For each fold, fine-tunes Chronos-2 with LoRA and generates predictions on the 6-month test window. Saves `fold_00.parquet` through `fold_16.parquet`.

**Runtime:** Several hours on CPU (each fold = 200 gradient steps on 2 years of data).

### Step 3a — A/B evaluation (fine-tuned predictions)
```
python scripts/03_run_ab_finetuned.py --run-dir outputs/<run_dir>
```
Replays all 17 folds' predictions through 6 sizing schemes. Prints performance table.

### Step 3b — A/B evaluation (zero-shot baseline)
```
python scripts/04_run_ab_zero_shot.py --run-dir outputs/<run_dir>
```
Same A/B harness but runs live Chronos inference on every test day (no fine-tuning). Provides the baseline: does fine-tuning actually help?

### Step 4 — SPY-default overlay
```
python scripts/05_run_spy_default_simulation.py --run-dir outputs/<run_dir>
```
Sweeps SPY-overlay configurations and compares against SPY buy-and-hold.

---

## 7.2 — Output Table Format

```
Scheme       Ann Ret     Sharpe    MaxDD   Hit Rate      N
──────────────────────────────────────────────────────────
OLD           +4.21%       0.61   -18.3%      0.523    847
NEW           +6.84%       0.92   -12.1%      0.558    601
NEW_CAP       +7.12%       1.04   -10.8%      0.561    601
DET           +5.43%       0.78   -14.2%      0.541    712
ENS_VETO      +8.21%       1.18    -9.3%      0.582    388
ENS_AVG       +7.98%       1.15    -9.7%      0.571    512
```

**Reading the table:**
- **ENS_VETO** typically has the fewest trades (N) but the highest hit rate — it only trades when both signals agree
- **OLD** has the most trades — the probability-only logic has no vol floor or consensus gate
- **Sharpe > 1.0** is the target for a strategy worth deploying

---

# Part 8: Key Correctness Invariants

A summary of the five critical correctness fixes in the codebase. Each one prevents a specific type of look-ahead bias or data contamination.

| Code | Fix | Where | What Goes Wrong Without It |
|------|-----|--------|---------------------------|
| **H1** | Unified Z-score on full crack series | `build_dataset.py` | Per-fold Z-score resets make +$40 crack look "normal" in one fold and "extreme" in another |
| **H3** | Per-fold seed: `42 + fold_idx` | `walkforward.py` | Different runs produce different LoRA weights → results not reproducible |
| **H4** | `effective_size` for PnL & hit rate | `ab_runner.py`, `metrics.py` | Counting today's decision against today's return → measuring decisions that haven't executed yet |
| **H5** | Rolling beta lagged by 1 day | `build_dataset.py` | Using today's return to compute today's beta → beta already "knows" the return it is hedging |
| **Strict <** | `history = df[df.index < T]` | `ab_runner.py`, `walkforward.py` | Current day's data leaks into the prediction for that same day |

---

# Appendix: Repository Structure

```
refiner/refiner_strategy/
├── refiner_strategy/               ← Library (never run directly)
│   ├── config.py                   ← All constants: tickers, weights, windows, thresholds
│   ├── data/
│   │   ├── futures_loader.py       ← CL/RB/HO download + 3:2:1 crack spread
│   │   └── build_dataset.py        ← Master dataset: crack Z, rolling beta, hedged returns
│   ├── signals/
│   │   └── det_signal.py           ← SMA crossover + 2-day persistence filter
│   ├── sizing/
│   │   └── schemes.py              ← 6 sizing functions (OLD → ENS_AVG)
│   ├── finetune/
│   │   └── walkforward.py          ← 17-fold purged walk-forward LoRA loop
│   ├── evaluation/
│   │   ├── ab_runner.py            ← A/B backtest harness (zero-shot + replay modes)
│   │   ├── metrics.py              ← Sharpe, drawdown, hit rate
│   │   └── spy_default_simulator.py ← SPY overlay sweep
│   └── utils/
│       └── torch_helpers.py        ← Device selection (CPU / CUDA / MPS)
├── scripts/                        ← Entry points (run these)
│   ├── 01_build_datasets.py        ← Fetch data, build master.csv
│   ├── 02_run_finetune_walkforward.py  ← Run 17-fold LoRA training
│   ├── 03_run_ab_finetuned.py      ← A/B test with saved predictions
│   ├── 04_run_ab_zero_shot.py      ← A/B test with live Chronos inference
│   └── 05_run_spy_default_simulation.py  ← SPY overlay benchmark
├── tests/
│   ├── smoke_chronos.py            ← Verify Chronos-2 loads and predicts
│   ├── smoke_fit.py                ← Verify LoRA fine-tuning API
│   ├── test_sizing.py              ← Unit tests: all 6 sizing schemes
│   ├── verify_pipeline.py          ← 20-day synthetic backtest
│   └── inspect_preds.py            ← Diagnostic: check prediction parquets
└── docs/
    └── Refiner_Strategy_Report.md  ← This report
```

---

*Report generated for Quant Management & Energy Trading Desk — May 2026*
