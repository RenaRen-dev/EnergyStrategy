# AI Refiner Backtest 
**Author:** Xinyi Ren | **Date:** May 2026

---

# Part 1: Strategy Overview

---

## 1.1 — The Core Idea

**DET:**
- **Daily execution:** Crack spread is positive $\rightarrow$ long (size = 100%)
- **Returns:** Accumulate daily returns
- **Friction:** Subtract transaction costs
- **Idle capital:** Unallocated capital is invested in SPY

**Different Models:**
- **Daily execution:** Crack spread + AI signal $\rightarrow$ long/short (dynamic sizing = ?)
- **Returns:** Accumulate daily returns
- **Friction:** Subtract transaction costs
- **Idle capital:** Unallocated capital is invested in SPY

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

**How DET Uses These Weights**

Example: If notional = $100 and DET signal = +1 (bullish):
- `position[VLO]` = +1 × ($100 × 0.25) = +$25  ← always this much when signal is +1
- `position[MPC]` = +1 × ($100 × 0.25) = +$25   
- `position[PSX]` = +1 × ($100 × 0.25) = +$25

**How AI Models Use These Weights**

Same example, with AI model:  
- If Chronos predicts: q50 = +0.5%, q90 = +1.0%, q10 = −0.1%
- Then: forecast_vol = 0.39%, edge = 1.28%, raw_size = 0.8x (clipped = 0.8)
- So: `position[VLO]` = 0.8 × ($100 × 0.25) = +$20 (less than DET's +$25)

---

# Part 2: Data Infrastructure

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

# Part 5: Determine the 6 strategies holding size everyday (The 6 Sizing Schemes)

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

**1. Ticker Friction (all individual stock position changes)**
```text
ticker_cost = Σ |Δposition_ticker| × (bps_per_leg / 10,000)
```
For each ticker $t$, sum the absolute change: `|target_pos_frac[t] - prev_pos_frac[t]|`

**2. SPY Friction (the macro allocation lever)**
```text
spy_cost = |Δspy_weight| × (bps_per_leg / 10,000)
```
Where `spy_weight = 1 - |net_refiner_allocation|`

**3. Borrow Cost (on yesterday's shorts, daily accrual)**
```text
borrow = Σ max(0, -position_t) × (borrow_bps_per_year / 10,000) / 252
```
Sum all negative positions, annualize the rate, and accrue daily.

**4. Total Daily Friction:**
```text
net_return = gross_return - ticker_cost - spy_cost - borrow
```


---

# Part 6: The SPY-Default Strategy


On each trading day, the portfolio is split between the refiner overlay and passive SPY:

```
alloc = net refiner allocation (between -1.0 and +1.0)

Portfolio Return = (1 - |alloc|) × SPY_Return  +  alloc × Refiner_Return
                   └── passive ──┘                └── active overlay ──┘
```



---



# Part 7: Running the Strategy

*How to execute the pipeline from data to results.*

---



## 7.2 — Output Table Format

```text
Scheme   TradeCost(bps) BorrowCost Ann Ret   Sharpe    MaxDD     Edge
--------------------------------------------------------------------
DET                10        0    +20.09%     0.70   -50.4%    +6.88pp
DET                10       50    +19.94%     0.69   -50.4%    +6.73pp
DET                15        0    +17.37%     0.60   -52.0%    +4.16pp
DET                15       50    +17.22%     0.60   -52.1%    +4.01pp
DET                20        0    +14.65%     0.51   -53.9%    +1.44pp
DET                20       50    +14.50%     0.50   -54.0%    +1.29pp
DET                25        0    +11.93%     0.41   -55.7%    -1.28pp
DET                25       50    +11.78%     0.41   -55.8%    -1.43pp
ENS_VETO           10        0    +22.73%     1.01   -34.8%    +9.52pp
ENS_VETO           10       50    +22.65%     1.00   -34.9%    +9.44pp
ENS_VETO           15        0    +20.06%     0.89   -37.2%    +6.86pp
ENS_VETO           15       50    +19.98%     0.88   -37.2%    +6.78pp
ENS_VETO           20        0    +17.40%     0.77   -39.5%    +4.19pp
ENS_VETO           20       50    +17.32%     0.77   -39.6%    +4.11pp
ENS_VETO           25        0    +14.73%     0.65   -41.8%    +1.53pp
ENS_VETO           25       50    +14.65%     0.65   -41.8%    +1.44pp
ENS_AVG            10        0    +26.64%     1.10   -30.2%   +13.44pp
ENS_AVG            10       50    +26.51%     1.09   -30.3%   +13.31pp
ENS_AVG            15        0    +23.25%     0.96   -32.5%   +10.04pp
ENS_AVG            15       50    +23.12%     0.95   -32.6%    +9.91pp
ENS_AVG            20        0    +19.85%     0.82   -34.8%    +6.65pp
ENS_AVG            20       50    +19.73%     0.81   -34.9%    +6.52pp
ENS_AVG            25        0    +16.46%     0.68   -36.9%    +3.25pp
ENS_AVG            25       50    +16.33%     0.67   -37.0%    +3.12pp
NEW_CAP            10        0    +17.22%     0.68   -37.8%    +4.02pp
NEW_CAP            10       50    +17.03%     0.67   -37.9%    +3.83pp
NEW_CAP            15        0    +13.37%     0.53   -43.8%    +0.16pp
NEW_CAP            15       50    +13.18%     0.52   -44.3%    -0.03pp
NEW_CAP            20        0     +9.52%     0.37   -52.7%    -3.69pp
NEW_CAP            20       50     +9.33%     0.37   -53.1%    -3.88pp
NEW_CAP            25        0     +5.66%     0.22   -60.1%    -7.55pp
NEW_CAP            25       50     +5.47%     0.22   -60.5%    -7.74pp
```

**Reading the table:**
- **ENS_AVG** achieves the highest overall returns and Sharpe ratio across all transaction cost regimes.
- **ENS_VETO** provides the second best performance, maintaining strong Sharpe ratios and edges over the SPY benchmark.
- **Sharpe > 1.0** is achieved in the lowest transaction cost tiers (10 bps) for both ensemble strategies, which is the target for a strategy worth deploying.
- Increasing **BPS RT** heavily impacts returns, indicating the strategy is sensitive to trading friction.
- Adding **Borrow** costs (50 bps) only marginally reduces the annualized returns, showing short-selling borrow fees are not the primary drag on performance.

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
