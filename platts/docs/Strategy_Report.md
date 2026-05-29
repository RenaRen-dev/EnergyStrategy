# Platts → CME Futures Strategy — v0.1 / v0.2 Report

**Status:** v0.1 backtest complete + v0.2 SPY-default capital overlay added; awaiting leakage validation
**Date:** 2026-05-28
**Y target:** RB (RBOB Gasoline) — Continuous 3rd-prompt futures
**X source:** PLATTS physical-market Z-scores (Snowflake share, normalized locally)
**Forecaster:** Chronos-2 (LoRA fine-tuned, 7-fold purged walk-forward)
**Window:** 2015-02-02 → 2021-12-31 (OOS: 2018-04-01 → 2021-12-31, ~880 trading days)

---

## 1. Executive Summary

The strategy uses PLATTS daily gasoline assessments — chosen by MOIRAI's attention-based feature selector — to forecast next-day RB futures returns via fine-tuned Chronos-2, traded through six sizing schemes against RB-BH and SPY-BH benchmarks.


**Framework**

DM213IA_Z (MOIRAI)→ SMA crossover → det_sig ∈ {-1, 0, +1}
                                      ↓ 
                              Position size in RB futures
                                      ↓
                              RB P&L on trading days
                              




**SPY-DEFAULT OVERLAY RESULTS (unused capital earns SPY)**

| Scheme | bps | Borrow | Ann Ret | Ann Vol | Sharpe | Max DD | Hit Rt | Trades | Edge |
|---|---|---|---|---|---|---|---|---|---|
| OLD | 1 | 0 | +27.76% | 24.51% | +1.133 | -29.42% | 56.90% | 186 | +8.36pp |
| OLD | 1 | 50 | +27.75% | 24.51% | +1.132 | -29.43% | 56.90% | 186 | +8.36pp |
| OLD | 5 | 0 | +26.23% | 24.50% | +1.071 | -29.52% | 56.90% | 186 | +6.84pp |
| OLD | 5 | 50 | +26.23% | 24.50% | +1.070 | -29.53% | 56.90% | 186 | +6.83pp |
| OLD | 10 | 0 | +24.33% | 24.50% | +0.993 | -29.65% | 56.90% | 186 | +4.94pp |
| OLD | 10 | 50 | +24.32% | 24.50% | +0.993 | -29.66% | 56.90% | 186 | +4.93pp |
| NEW | 1 | 0 | +31.01% | 44.08% | +0.704 | -51.99% | 53.65% | 154 | +11.62pp |
| NEW | 1 | 50 | +30.94% | 44.08% | +0.702 | -52.03% | 53.65% | 154 | +11.55pp |
| NEW | 5 | 0 | +29.68% | 44.08% | +0.673 | -52.86% | 53.65% | 154 | +10.29pp |
| NEW | 5 | 50 | +29.61% | 44.08% | +0.672 | -52.89% | 53.65% | 154 | +10.22pp |
| NEW | 10 | 0 | +28.02% | 44.08% | +0.636 | -53.91% | 53.65% | 154 | +8.63pp |
| NEW | 10 | 50 | +27.95% | 44.08% | +0.634 | -53.94% | 53.65% | 154 | +8.56pp |
| NEW_CAP | 1 | 0 | +32.99% | 37.87% | +0.871 | -51.99% | 53.65% | 190 | +13.60pp |
| NEW_CAP | 1 | 50 | +32.93% | 37.87% | +0.869 | -52.03% | 53.65% | 190 | +13.53pp |
| NEW_CAP | 5 | 0 | +31.73% | 37.87% | +0.838 | -52.86% | 53.65% | 190 | +12.34pp |
| NEW_CAP | 5 | 50 | +31.67% | 37.87% | +0.836 | -52.89% | 53.65% | 190 | +12.28pp |
| NEW_CAP | 10 | 0 | +30.16% | 37.88% | +0.796 | -53.91% | 53.65% | 190 | +10.77pp |
| NEW_CAP | 10 | 50 | +30.09% | 37.88% | +0.795 | -53.94% | 53.65% | 190 | +10.70pp |
| **DET** | **1** | **0** | **+48.45%** | **36.56%** | **+1.325** | **-40.47%** | **55.38%** | **246** | **+29.05pp** |
| **DET** | **1** | **50** | **+48.29%** | **36.56%** | **+1.321** | **-40.56%** | **55.38%** | **246** | **+28.90pp** |
| DET | 5 | 0 | +46.26% | 36.56% | +1.265 | -41.80% | 55.38% | 246 | +26.87pp |
| DET | 5 | 50 | +46.11% | 36.56% | +1.261 | -41.90% | 55.38% | 246 | +26.72pp |
| DET | 10 | 0 | +43.54% | 36.58% | +1.190 | -43.63% | 55.38% | 246 | +24.15pp |
| DET | 10 | 50 | +43.39% | 36.58% | +1.186 | -43.73% | 55.38% | 246 | +24.00pp |
| ENS_VETO | 1 | 0 | +43.75% | 32.73% | +1.337 | -35.62% | 56.40% | 234 | +24.36pp |
| ENS_VETO | 1 | 50 | +43.71% | 32.73% | +1.336 | -35.66% | 56.40% | 234 | +24.32pp |
| ENS_VETO | 5 | 0 | +41.73% | 32.73% | +1.275 | -37.37% | 56.40% | 234 | +22.34pp |
| ENS_VETO | 5 | 50 | +41.69% | 32.73% | +1.274 | -37.41% | 56.40% | 234 | +22.30pp |
| ENS_VETO | 10 | 0 | +39.20% | 32.73% | +1.197 | -39.50% | 56.40% | 234 | +19.81pp |
| ENS_VETO | 10 | 50 | +39.16% | 32.73% | +1.196 | -39.54% | 56.40% | 234 | +19.77pp |
| ENS_AVG | 1 | 0 | +38.80% | 34.41% | +1.128 | -52.29% | 56.11% | 233 | +19.41pp |
| ENS_AVG | 1 | 50 | +38.75% | 34.41% | +1.126 | -52.33% | 56.11% | 233 | +19.36pp |
| ENS_AVG | 5 | 0 | +36.91% | 34.41% | +1.073 | -53.51% | 56.11% | 233 | +17.51pp |
| ENS_AVG | 5 | 50 | +36.85% | 34.41% | +1.071 | -53.54% | 56.11% | 233 | +17.46pp |
| ENS_AVG | 10 | 0 | +34.54% | 34.41% | +1.004 | -54.98% | 56.11% | 233 | +15.15pp |
| ENS_AVG | 10 | 50 | +34.49% | 34.41% | +1.002 | -55.02% | 56.11% | 233 | +15.09pp |

**SPY baseline:** Ann Ret = +19.39%, Sharpe = +0.929, Max DD = -33.72%



**BACKTEST @ 5.0 bps round-trip**

| Scheme | Sharpe | Ann Return | Ann Vol | Max DD | Hit Rate | Trades | Txn $ |
|---|---|---|---|---|---|---|---|
| OLD | +0.620 | +9.16% | 14.77% | -13.11% | 56.90% | 186 | $7.22 |
| NEW | +0.525 | +23.61% | 44.96% | -80.23% | 53.64% | 154 | $10.12 |
| NEW_CAP | +0.567 | +21.68% | 38.25% | -80.23% | 53.64% | 190 | $9.51 |
| **DET** | **+0.948** | **+33.69%** | **35.56%** | **-54.37%** | **55.24%** | **244** | **$10.32** |
| ENS_VETO | +0.853 | +26.12% | 30.63% | -47.95% | 56.30% | 234 | $9.55 |
| ENS_AVG | +0.850 | +28.86% | 33.95% | -69.17% | 56.03% | 233 | $9.74 |
| RB_BH | +0.084 | +3.82% | 45.49% | -141.42% | n/a | 0 | $0.00 |
| SPY_BH | +0.819 | +17.16% | 20.96% | -41.12% | n/a | 0 | $0.00 |





**MOIRAI Top-5 selected (used as X covariates):**

| Rank | Symbol | Attention | Product | Grade |
|---|---|---|---|---|
| 1 | DM213IA | 0.110 | Unleaded Gasoline | Midgrade (Branded) |
| 2 | DM216IA | 0.105 | Unleaded Gasoline | Midgrade (Branded) |
| 3 | DM173ZY | 0.081 | Unleaded Gasoline | Midgrade (Unbranded) |
| 4 | DP173ZY | 0.081 | Unleaded Gasoline | Premium (Unbranded) |
| 5 | DM214IA | 0.055 | Unleaded Gasoline | Midgrade (Branded) |


**Train Chronors-2 using these Top-5 selected**
**Do strategy only use Top-1**



---

## 2. Economic Premise

**The thesis:** PLATTS gasoline physical-market assessments capture supply-side micro-information (regional inventories, blending economics, transport spreads) that the CME RB futures market has not yet fully priced by its 2:30 PM ET settle.

**The mechanism:** RB futures price the RBOB *blendstock*; PLATTS DM-series symbols price *finished gasoline* (RBOB + ethanol + blending margin + retail markup). The relationship is mechanical but lossy: changes in the physical complex (refinery outages, regional grade spreads, Gulf Coast vs. Harbor basis) propagate into RB but with frictional delay.

**Honest comparison to refiner strategy:**
- Refiner had a *mechanical identity*: `refiner_profit ≡ crack_spread`. Predicting crack predicts refiners by accounting.
- This strategy has *no identity*. The PLATTS → RB linkage is statistical, regime-dependent, and may decay.
- The edge is real but weaker. The strategy is best deployed as a sizing/filter layer atop a directional view, not as a standalone alpha.

---

## 3. Data Pipeline

### 3.1 Source
- **Raw:** `SPGE_MARKETDATA_SHARE.MDV2.PRICEDATA` (Snowflake share, read-only)
- **Date range downloaded:** 2015-02-01 → 2022-09-01 (16 chunks, ~625 MB compressed)
- **Schema:** SYMBOL, MDC, DESCRIPTION, ASSESSDATE, VALUE, UOM, CURRENCY, BATE, ISCORRECTED

### 3.2 Local processing (ported from `commodity/` to pandas)
Stages run end-to-end on the user's MacBook, no Snowflake writes:

1. **Parse DESCRIPTION** — regex parser from `commodity/utility/parse_description_udf.py` adds PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING, IS_SPOT.
2. **FX normalization** — FRED rates (DEXCAUS, DEXUSEU, EXGEUS for pre-1999 Euro) converted all CURRENCY to USD.
3. **UOM normalization** — converted to USD/GAL via specific-gravity lookup (gasoline 8.5 bbl/MT, distillates 7.45, heavy oils 6.3).
4. **Z-score normalization** — per-symbol dense daily grid + ffill + 256-day rolling Z clipped to [-3, +3].

### 3.3 RB Y target
- 91 per-contract CSVs in `platts/data/train/futures/RB_*.csv`
- 3rd-prompt continuous series (offline-only, no yfinance fallback)
- Log return: `R_t = ln(S_t / S_{t-1})`
- 1,744 trading days, settlement range $1.69–$2.33/gal

---

## 4. Feature Selection — MOIRAI Discovery

### 4.1 Process
- Loaded 47,639 PLATTS symbols → filtered by gasoline/distillate/crude product whitelist → 16,847 symbols
- Liquidity filter (≥80% coverage on dense grid) → 9,739 symbols
- Variance-rank → top 500 candidates
- MOIRAI batched scans (19 PLATTS + RB anchor per batch, 27 batches)
- Captured last-layer encoder cross-attention; read attention **from** RB query tokens **to** each PLATTS variate's key/value tokens
- Higher attention = stronger PLATTS → RB conditional dependence

### 4.2 Top-5 selected (used as X covariates)

| Rank | Symbol | Attention | Product | Grade |
|---|---|---|---|---|
| 1 | DM213IA | 0.110 | Unleaded Gasoline | Midgrade (Branded) |
| 2 | DM216IA | 0.105 | Unleaded Gasoline | Midgrade (Branded) |
| 3 | DM173ZY | 0.081 | Unleaded Gasoline | Midgrade (Unbranded) |
| 4 | DP173ZY | 0.081 | Unleaded Gasoline | Premium (Unbranded) |
| 5 | DM214IA | 0.055 | Unleaded Gasoline | Midgrade (Branded) |

**Observation: all 5 are gasoline grade variants** (4 of 5 are midgrade). MOIRAI converged on a single product category. This is the intended outcome (RB is gasoline) but raises a redundancy concern (Section 8.4).

---

## 5. Model — Chronos-2 Walk-Forward Fine-Tune

### 5.1 Geometry (mirrors refiner)
- 7 non-overlapping folds, 6-month test windows
- 24-month rolling training window per fold
- 5-day purge gap between train_end and test_start (prevents leakage at fold boundary)
- LoRA seed: `LORA_BASE_SEED + fold_idx` (deterministic per fold)
- OOS test span: 2018-04-01 → 2021-12-31

### 5.2 Per-fold mechanics
- **Input:** 6 variates → variate 0 = `RB_LogReturn × 100`, variates 1–5 = top-5 PLATTS Z-scores
- **Fine-tune:** 200 steps, batch 64, learning rate 1e-5
- **Inference:** for each test day T, strict `history[history.index < T]`, generate quantile forecast q10..q90 + p_up
- **Output:** `chronos_predictions/fold_NN.parquet` → concatenated as `all_preds.parquet`

### 5.3 Device note
Apple MPS doesn't support float64 (required by MOIRAI and parts of Chronos's normalization); script auto-selects CPU. ~90 minutes total on Apple Silicon for full 7-fold run.

---

## 6. Backtest — A/B Harness

### 6.1 Sizing schemes (six, ported from refiner with single-asset adaptation)
| Scheme | Logic |
|---|---|
| OLD | Probability deadband on `p_up`, no vol awareness (legacy) |
| NEW | Vol-targeted with q50/p_up consensus gate |
| NEW_CAP | NEW + 20-day realized-vol cap at 4% |
| DET | Pure PLATTS-driver SMA crossover (10-day SMA, 2-day persistence) on top MOIRAI symbol |
| ENS_VETO | NEW_CAP gated by DET sign agreement |
| ENS_AVG | Bates-Granger combination of Chronos q50 and DET |

### 6.2 Calibration deltas vs. refiner
RB raw log returns are ~10× larger than refiner's hedged equity returns, so:
- `TARGET_DAILY_VOL`: 0.015 (was 0.01)
- `VOL_FLOOR`: 0.005
- `VOL_CAP`: 0.04 (was 0.02)
- `DET_SIGNAL_MAG`: 0.02 (was 0.005)

### 6.3 Accounting (single-asset)
- One product (RB), notional = $100 per scheme
- No basket, no SPY hedge (the asset *is* the future, not equity)
- Friction: `bps × |Δposition|` per day
- PnL: `position_{t-1} × return_t − friction`
- T+1 convention: today's signal used at today's close to size for tomorrow

### 6.4 Benchmarks
- **RB_BH** — long-only RB at full notional, no friction
- **SPY_BH** — long-only S&P 500 (from `platts/data/train/SPY_daily.csv`), no friction

---

## 8. Critical Concerns

### 8.1 Data leakage (highest priority)

**The question:** Do PLATTS daily assessments incorporate the RB 2:30 PM ET settle that occurred earlier the same day?

**Timing chain that would constitute leakage:**
```
14:30 ET    RB futures settle
15:00–17:00 ET  PLATTS dealers update quotes, possibly using RB settle
16:00–17:00 ET  PLATTS publishes "end-of-day" DM-series assessments
```

If true, then `PLATTS(T-1) Z-score` already contains `RB(T-1)` price information, and using it to forecast `RB(T)` is not capturing PLATTS → RB lead-lag — it is exploiting the autoregression of RB itself through a lagged proxy.

**Symptoms consistent with leakage:**
- DET alone outperforms ENS schemes that include Chronos → the SMA crossover is so strong it suggests near-perfect information
- All top-5 MOIRAI drivers are gasoline (perfectly correlated with RB) — the model picked the most rb-shaped signal available
- Hit rates above 55% on a futures market that should be nearly efficient

**Symptoms inconsistent with leakage:**
- Hit rates *not* near 100% (would be the smoking gun)
- DET trades infrequently — pure autoregression wouldn't gate trades like this
- Different schemes produce meaningfully different Sharpes — pure leakage would make all schemes equally great

**Verdict:** Leakage cannot be ruled out from the current backtest. **The strategy should not be deployed before validation.**

### 8.2 Transaction costs are optimistic
- 1 bps RT was used in the headline. Realistic RT for retail-size RB positions is 5–10 bps after spreads + slippage.
- DET scheme trades 244 times in ~880 days → frequent rebalancing
- At 10 bps, expected return drop of 30–50%
- ENS_VETO is more robust to costs (fewer trades) and remains the best cost-adjusted candidate

### 8.3 The 2015–2021 window is atypical
Includes:
- 2015–2016 oil crash (WTI from $107 to $26)
- 2020 COVID (RB futures briefly negative)
- 2021 supply-shortage rally

These are exactly the regimes where lagged-information edges thrive. The strategy may not generalize to calmer markets.

### 8.4 Covariate redundancy
The 5 MOIRAI-selected symbols are all gasoline grade variants — likely 0.95+ correlated with each other. Effective covariate count is closer to 1, not 5. Chronos is essentially seeing `(RB_return, gasoline_z)` as a bivariate problem. Adding more covariates likely will not help; the next experiment should add a *different product category* (e.g., crude WTI Z-score) for genuine information.

### 8.5 No basis risk modeling
PLATTS DM213IA prices Gulf Coast finished gasoline. RB futures price NY Harbor RBOB blendstock. Their basis varies substantially (regional supply, ethanol prices, branded premium). The strategy assumes their Z-scores co-move with RB; basis-shift periods could blow this up.

### 8.6 No live validation
Same caveat as refiner: backtest != production. Real execution will reveal additional issues (fill quality, intraday revisions to PLATTS, model deployment latency).

---

## 9. Validation Tests Required Before Deployment

### Test 1 — Pre-settle PLATTS filter
Re-run the entire pipeline with PLATTS data filtered to assessments published **before 13:00 ET** (90 minutes before RB settle). This requires hour-granularity ASSESSDATE; if only date-granularity is available, this test is impossible and Test 2 becomes mandatory.
- **Pass:** DET return ≥ 25% (edge holds without potential post-settle info)
- **Fail:** DET return collapses below 15% (edge was post-settle artifact)

### Test 2 — Out-of-sample window
Download 2022-01-01 → 2026-05-01 PLATTS data and rerun. The 2022+ era post-dates the model's training distribution and reflects more mature pricing infrastructure.
- **Pass:** DET return ≥ 15% (edge generalizes)
- **Fail:** DET return < 5% (edge was 2015–2021 specific)

### Test 3 — Univariate Chronos baseline
Re-run step 6 with **no PLATTS covariates** — only `RB_LogReturn` as the single variate. If univariate Chronos matches multivariate Chronos's Sharpe, then PLATTS adds no value; we're just doing autoregression.
- **Pass:** Multivariate Sharpe > univariate by ≥0.15
- **Fail:** Sharpes within 0.05 of each other

### Test 4 — Random-symbol placebo
Re-run step 4 (MOIRAI) but **randomly select** 5 PLATTS symbols instead of the top-5. If the random covariates produce similar backtest results, MOIRAI's selection was not adding value.
- **Pass:** Top-5 outperforms random-5 by ≥3pp annualized
- **Fail:** Random-5 performs within 1pp of top-5

### Test 5 — Cost stress
Re-run step 8 at 5, 10, 20, 50 bps RT. The strategy is viable only if it maintains positive edge over SPY at the realistic 5–10 bps level.

---

## 10. What We Built — Code Inventory

```
platts/
├── README.md
├── docs/
│   ├── Implementation_Plan.md
│   └── Strategy_Report.md          (this file)
├── data/                            (gitignored — keys, train CSVs)
│   ├── xren_private_key.p8
│   ├── train/futures/RB_*.csv       (91 contracts)
│   └── train/SPY_daily.csv
├── scripts/
│   ├── 01_download_pricedata.py     Snowflake → chunked parquet
│   ├── 02_process_pricedata.py      parse → fx → normalize → zscore
│   ├── 03_build_rb_returns.py       3rd-prompt RB return series
│   ├── 04_moirai_rb_discovery.py    MOIRAI RB-anchored attention scan
│   ├── 05_build_master_dataset.py   align top-K + RB into one parquet
│   ├── 06_chronos_walkforward.py    7-fold LoRA WFO
│   ├── 07_build_det_signal.py       PLATTS SMA crossover DET signal
│   ├── 08_run_ab_backtest.py        6-scheme A/B harness, SPY/RB benchmarks
│   └── 09_run_spy_default_simulation.py   capital-efficient overlay
└── outputs/                         (gitignored — all runtime artifacts)
    ├── pricedata/{raw,parsed,normalized,zscore}/
    ├── rb_returns.parquet
    ├── moirai_rb_ranking.csv
    ├── master_dataset.parquet
    ├── chronos_predictions/
    ├── platts_det_signal.parquet
    ├── backtest/{results,pnl_curves}_<bps>bps.csv              (v0.1)
    └── backtest_spy_default/{summary,nav_*}.csv                (v0.2)
```

---

## 11. Honest Limitations

1. **Data leakage is unverified.** Section 8.1 covers the strongest case for caution.
2. **1 bps RT is unrealistic.** Real costs likely 5–10 bps; metrics need to be re-evaluated at those levels.
3. **Single window (2015–2021).** Atypical regime; no out-of-sample validation on 2022+ data yet.
4. **Covariate redundancy.** Top-5 are nearly co-linear; effective dimensionality is closer to 1.
5. **No live trading validation.** Backtest assumptions about fill quality, slippage, and intraday revisions are unverified.
6. **PLATTS Gulf Coast vs. RB NY Harbor.** Basis risk not modeled; regional spreads could break the relationship.
7. **No survivorship-bias audit.** PLATTS symbols that delisted during the window were dropped via the 80% coverage filter; their absence is a small but unmeasured bias.
8. **Chronos adds marginal value.** DET alone has the highest Sharpe; the AI component is justified mainly by ensemble smoothing, not standalone forecasting power.

---

## 12. Recommendations

### Before any deployment
1. **Run Test 2 (2022–2026 OOS).** This is the cheapest, highest-information validation. If DET return collapses below 15%, the strategy is dead.
2. **Run Test 1 (pre-settle filter)** if ASSESSDATE has hour granularity.
3. **Run Test 3 (univariate baseline)** to confirm PLATTS data is actually contributing.

### If validation passes
- **Deploy ENS_VETO** (not DET standalone) — better risk-adjusted, more robust to costs, fewer trades.
- **Real costs:** Stress-test at 10 bps; assume Sharpe drops by ~0.2–0.3 from the headline.
- **Position sizing:** Cap notional at 25% of available capital; treat as a *satellite* allocation around an SPY core (per the SPY-default overlay logic in step 9).

### If validation fails
- The pipeline (steps 1–6) is still reusable infrastructure.
- Pivot the strategy: try HO (heating oil) or CL (crude) as Y targets. PLATTS coverage of distillate and crude markets is broader and may show more diverse MOIRAI selections.
- Investigate spread trades (RB-CL, RB-HO) — basis spreads may be less efficient than outright price predictions.

---

## 13. Definition of "Done" for v0.1 + v0.2

| Criterion | Status |
|---|---|
| End-to-end pipeline runs (scripts 01–09) | ✅ |
| Master dataset assembled (1,744 rows, 7 columns) | ✅ |
| Chronos walk-forward predictions generated | ✅ |
| A/B backtest produces 6-scheme metrics | ✅ |
| SPY-default overlay simulation works | ✅ |
| At least one scheme beats SPY at 1 bps | ✅ (5 of 6) |
| At least one scheme beats SPY at 10 bps | **❓ (untested)** |
| Leakage risk audited and resolved | **❌** |
| OOS validation on 2022+ data | **❌** |

**v0.1 ships the bare backtest infrastructure. v0.2 adds the SPY-default capital overlay for fair benchmarking. v0.3 must include leakage validation (Tests 1–3) and 2022+ OOS testing before any production decision.**
