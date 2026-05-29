# Platts → CME Futures Strategy — Implementation Plan

**Status:** Draft v0.1 (planning only, no code written yet)
**Author:** Drafted with Claude, 2026-05-27
**Y target:** RB (then HO, CL, then spreads)
**X source:** PLATTS Z-scores in Snowflake, selected by MOIRAI
**Forecaster:** Chronos-2 (LoRA fine-tuned, walk-forward)
**Backtest:** Refiner A/B harness, adapted (Chronos + PLATTS-derived DET; no crack spread)

---

## 1. Why this project, what changes vs. refiner

The refiner strategy traded *equities* whose margins are mechanically tied to the crack spread. The DET signal was the crack-spread SMA crossover — a defensible rule because the crack spread literally **is** the refiners' margin.

This project trades the **futures themselves** (RB, HO, CL). There is no margin identity to anchor a DET rule on. Instead, we let MOIRAI's cross-attention discover which PLATTS physical-market products *lead* each future, and we build the DET from those.

| | Refiner | Platts (this project) |
|---|---|---|
| Universe | 7 refiner stocks | RB, HO, CL (one at a time) |
| Y target | Beta-hedged equity return | Future log-return, next settle-to-settle |
| X covariate to Chronos | Crack Z-score | Top-K PLATTS Z-scores (MOIRAI-ranked) |
| DET signal | 10-day SMA crossover of crack spread | SMA crossover (or sign) of top MOIRAI driver |
| Sizing schemes | OLD, NEW, NEW_CAP, DET, ENS_VETO, ENS_AVG | Same six, adapted to single-asset |
| Hedge | Rolling beta vs SPY | None (futures are directly tradable) |
| Benchmark | SPY buy-and-hold | RB buy-and-hold + cash |

---

## 2. The five steps end-to-end

```
[1] Snowflake-side filter  →  candidate parquet  (≤ 200 PLATTS symbols)
[2] MOIRAI sweep w/ RB as anchor  →  ranked X list  (top 5-20)
[3] Build master dataset  →  RB return + top-K Z-scores, aligned daily
[4] Chronos-2 walk-forward fine-tune  →  q10..q90 + p_up per test day
[5] A/B backtest (6 schemes) + PLATTS-DET signal  →  PnL, Sharpe, DD
```

Each step has a script in `scripts/` and writes to a per-run output folder.

---

## 3. PLATTS data: download and search strategy

> **The problem.** Snowflake holds ~163M rows across 6,345 PLATTS products. Pulling everything is wasteful (most products are illiquid or unrelated to RB) and makes MOIRAI scans impossible on 8GB VRAM.

**The principle: filter aggressively in Snowflake, scan locally with MOIRAI, pull the winners into Chronos.**

### 3.1 Three-stage funnel

| Stage | Input | Output | Where it runs | Estimated rows |
|---|---|---|---|---|
| **Pre-filter** | 163M rows, 6,345 products | ~200 candidate symbols × dates | Snowflake SQL | ~500K |
| **MOIRAI scan** | 200 candidates + RB anchor | Top-20 ranked X list | Local GPU | — |
| **Chronos covariates** | 5–20 selected X + RB return | Master parquet | Local CPU/GPU | ~3K rows × 20 cols |

### 3.2 Pre-filter SQL (runs in Snowflake, returns ~500K rows)

```sql
WITH liquid AS (
  SELECT SYMBOL,
         PRODUCT,
         COUNT(*)              AS n_obs,
         COUNT(*) / (DATEDIFF('day', MIN(ASSESSDATE), MAX(ASSESSDATE)) + 1.0)
                               AS coverage,
         VAR_SAMP(Z_SCORE)     AS z_var
  FROM CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY
  WHERE ASSESSDATE >= '2014-01-01'
    AND Z_SCORE IS NOT NULL
    AND PRODUCT IN (
      -- Gasoline-related (for RB):
      'Unleaded Gasoline', 'RBOB Gasoline', 'Naphtha', 'Alkylate',
      'Reformate', 'Octane', 'MTBE', 'Ethanol',
      -- Diesel/heating-related (for HO):
      'Heating Oil', 'Diesel', 'Gasoil', 'ULSD', 'Jet',
      -- Crude (for CL and cross-product context):
      'Brent', 'WTI', 'Dubai', 'Bonny Light', 'LLS',
      -- Cracks and refining margins (for context):
      'Gasoline Crack', 'Distillate Crack', '3:2:1 Crack'
    )
  GROUP BY SYMBOL, PRODUCT
  HAVING n_obs >= 1000           -- ~4+ years of trading days
     AND coverage >= 0.60        -- 60%+ of business days populated
     AND z_var > 0.05            -- not flatlined
)
SELECT s.SYMBOL, s.ASSESSDATE, s.Z_SCORE
FROM CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY s
JOIN liquid USING (SYMBOL)
WHERE s.ASSESSDATE >= '2014-01-01'
ORDER BY s.SYMBOL, s.ASSESSDATE;
```

Tunable knobs: product whitelist (start broad, prune after MOIRAI ranks), `n_obs` floor, `coverage` floor, `z_var` floor. All live in `config.py` so we can rerun without touching SQL.

### 3.3 MOIRAI scan with **RB as fixed anchor** (not free-discovered)

`commodity/ml/inference.py` runs MOIRAI to *discover* the most influential symbol globally. Here we already know what we want — RB — so we **fix the target** and ask "which of the 200 candidates most strongly attends to RB?"

Two implementation options:

**A. Drop RB into the candidate matrix and read attention into RB's column.** Cheaper, single-pass. Implementation: append the RB Z-score series to the pivoted candidate DataFrame, run MOIRAI's batched attention capture, sum attention weights *into* RB's variate ID. Symbols with the highest summed weight are the top X.

**B. Per-symbol pairwise MOIRAI runs.** Run MOIRAI on `[RB, candidate_i]` for every candidate. Cleaner influence signal but 200x more inferences. Use only if Option A's ranks are noisy.

Default to Option A.

### 3.4 Caching

Every Snowflake pull writes to `outputs/<run-id>/data/platts_candidates.parquet` keyed by `(product_whitelist_hash, date_range)`. Re-runs skip the pull if the cache exists. Set `--force-refresh` to override.

### 3.5 Incremental updates (post-MVP)

Daily refresh only needs new dates appended to the cached parquet — write a thin `update_candidates_to_today.py` once the MVP works.

---

## 4. Y construction: futures returns from train.zip

The user will place `train.zip` on disk. Expected contents (per the project brief): daily CME settlement prices for RB, HO, CL with date and settlement columns at minimum. Likely format (mirrors `refiner/refiner_strategy/data/futures/{PROD}_{YYYY}_{MM}.csv`).

### 4.1 Loader

`platts_strategy/data/futures_loader.py` — extends the refiner loader. Key changes:
- Output a **single product series** (no crack spread). Caller selects RB / HO / CL.
- Output **log return** column (`R_t = ln(S_t / S_{t-1})`) plus raw settlement.
- Same fixed-tenor logic (default: 3rd prompt) to avoid roll-day discontinuities.

### 4.2 Alignment

PLATTS Z-scores are in `Asia/UTC`-ish daily assessments; CME settles 2:30 PM ET. We align on **the PLATTS date that precedes the CME settle being predicted**. Concretely: `R_{t+1} = ln(S_{t+1} / S_t)` is predicted using PLATTS Z-scores observed at date `t` (or earlier).

### 4.3 Strict-< rule (preserved from refiner)

`history = master[master.index < T]`. Today's settle never enters today's forecast context.

---

## 5. Folder structure (to be created after plan approval)

```
platts/
├── README.md                        ✅ created
├── requirements.txt                  ← pinned subset of root requirements
├── docs/
│   ├── Implementation_Plan.md       ✅ this file
│   └── walkthrough.md                ← plain-English narrative once code lands
├── platts_strategy/                  ← importable package
│   ├── __init__.py
│   ├── config.py                     ← single source of truth (see §9)
│   ├── data/
│   │   ├── snowflake_loader.py       ← §3 pre-filter SQL + Arrow fetch + cache
│   │   ├── futures_loader.py         ← §4 train.zip loader
│   │   └── build_dataset.py          ← master DataFrame assembler
│   ├── discovery/
│   │   └── moirai_selector.py        ← §3.3 RB-anchored MOIRAI sweep
│   ├── signals/
│   │   └── platts_det_signal.py      ← §6 SMA crossover on top driver
│   ├── finetune/
│   │   └── walkforward.py            ← §7 adapted purged WFO
│   ├── sizing/
│   │   └── schemes.py                ← §8 six schemes for single-asset futures
│   ├── evaluation/
│   │   ├── ab_runner.py              ← §8 adapted A/B harness
│   │   └── metrics.py                ← reused from refiner
│   └── utils/
│       └── torch_helpers.py          ← copy from refiner
├── scripts/
│   ├── 01_extract_platts_candidates.py
│   ├── 02_run_moirai_discovery.py
│   ├── 03_build_master_dataset.py
│   ├── 04_run_finetune_walkforward.py
│   └── 05_run_ab_backtest.py
├── tests/
│   ├── test_snowflake_filter_sql.py
│   ├── test_futures_loader.py
│   ├── test_sizing.py
│   └── test_walkforward_boundaries.py
└── outputs/                          ← gitignored, created at runtime
    └── <run-id>/
        ├── data/
        │   ├── platts_candidates.parquet
        │   ├── moirai_ranking.csv
        │   └── master.parquet
        ├── predictions/
        │   └── fold_NN.parquet
        ├── backtest/
        │   └── results.csv
        └── run_config.json
```

Matches the refiner layout (`refiner/refiner_strategy/`) so a developer who knows refiner can navigate this immediately.

---

## 6. DET signal — what replaces the crack spread

The refiner DET was: 10-day SMA crossover of the 3:2:1 crack spread, 2-day persistence. Here we don't have an obvious accounting-identity signal. Three candidates, in order of preference:

### 6.1 Top-driver SMA crossover (default)

After MOIRAI ranks PLATTS symbols, take the **#1 driver** of RB. Compute its 10-day SMA on the Z-score series; signal +1 when above, −1 when below, with 2-day persistence. This is structurally identical to the refiner DET — same code, different input series.

### 6.2 Weighted-driver composite

Take the top-3 MOIRAI drivers, weight by their attention scores, and apply the same SMA crossover to the composite. More robust if any single driver is noisy.

### 6.3 MOIRAI-direct DET (alternative)

Use MOIRAI's own 1-day-ahead median forecast on RB's Z-score: sign = +1 if forecast > last actual, −1 if below. Skips the SMA step; lets the foundation model do its own crossover.

**Decision rule:** Default to **6.1**. If hit rate < 50% in a quick smoke test, fall back to **6.2**.

The DET signal is unlagged in the signal module; the A/B harness lags by 1 day at consumption time (mirrors refiner's `.shift(1)` convention).

---

## 7. Chronos-2 walk-forward fine-tune (adapted from refiner)

Reuse `refiner_strategy/finetune/walkforward.py` almost verbatim. Three changes:

### 7.1 Target and covariates

```python
# Refiner (current):
target  = train_data[f"{ticker}_Hedged_Return"] * 100
covar   = train_data["Crack_Z_Score"]
ctx     = np.stack([target, covar])             # shape (2, T)

# Platts (new):
target  = train_data[f"{PRODUCT}_LogReturn"] * 100
covars  = [train_data[c] for c in TOP_K_PLATTS] # K MOIRAI-selected
ctx     = np.stack([target, *covars])           # shape (1 + K, T)
```

Chronos-2 supports multi-variate context natively (`Variate 0 = target, Variate 1+ = covariates`). The refiner code already handles this; we just feed more covariates.

### 7.2 Loop is per-product, not per-ticker

Refiner loops over 7 tickers per fold. Platts loops over 1 product (RB first). Strictly simpler.

### 7.3 Fold geometry

Keep refiner's geometry: 24-month train, 6-month test, 5-day purge gap, `LORA_BASE_SEED + fold_idx` per fold. Adjust only `OOS_TEST_START` based on PLATTS history start (likely 2017–2018 once we know the train.zip coverage).

### 7.4 Output schema

Same as refiner — `Date, Product, q10, …, q90, p_up` parquet per fold. Replace `Ticker` column with `Product`.

---

## 8. A/B harness: six sizing schemes for single-asset futures

Reuse `refiner_strategy/evaluation/ab_runner.py`. Three changes:

### 8.1 Single-asset accounting

Drop the basket loop. Drop the AB_WEIGHTS dict. Positions live on one product at a time. The `_step_one_day` inner loop collapses from `for ticker in basket` → just one product.

### 8.2 No hedge

Remove `_rolling_beta_lagged` and `*_Hedged_Return` columns. Y is the raw futures log-return; the sizer's output is dollar notional in the future directly.

### 8.3 Sizing schemes — adapt, don't replace

| Scheme | Refiner | Platts adaptation |
|---|---|---|
| OLD | Probability deadband on p_up | **Identical.** Pure Chronos p_up sizer with [0.40, 0.60] deadband. |
| NEW | Vol-targeted with consensus gate (q50 vs p_up) | **Identical.** |
| NEW_CAP | NEW + realized-vol cap | **Identical.** |
| DET | Crack-SMA signal × notional | **PLATTS-driver DET** × notional (§6.1). |
| ENS_VETO | Chronos NEW_CAP gated by DET agreement | **Identical** in logic; uses §6.1 DET. |
| ENS_AVG | Bates-Granger combination of q50 and DET | **Identical** in logic; uses §6.1 DET. |

`DET_SIGNAL_MAG` is recalibrated to RB's typical daily return scale (~1.5–2.5%). The refiner value (0.005 = 50 bps) is set for hedged equity returns; for raw futures we'd expect ~0.02. Tune empirically.

### 8.4 Benchmarks

Three lines on every PnL chart:
1. The sizing scheme
2. **Long-only RB buy-and-hold** (equivalent of SPY benchmark in refiner)
3. **Cash** (zero PnL line, for floor)

### 8.5 Transaction costs

Refiner uses 10 bps round-trip on equity. CME futures are cheaper — 1 bp RT is realistic for RB/HO/CL at exchange-cleared sizes. Plan: sweep `[1, 5, 10]` bps so we see the cost-sensitivity curve.

---

## 9. config.py — single source of truth

Mirror refiner's `config.py` style. New / changed knobs:

```python
# Product universe
PRODUCT          = "RB"                # then "HO", "CL"
TENOR_PROMPT     = 3                   # 3rd-prompt fixed-tenor

# PLATTS pre-filter
PLATTS_PRODUCTS  = (...gasoline/distillate/crude list per §3.2...)
PLATTS_MIN_OBS   = 1000
PLATTS_MIN_COV   = 0.60
PLATTS_MIN_VAR   = 0.05

# MOIRAI discovery
MOIRAI_MODEL_ID  = "Salesforce/moirai-1.1-R-small"
MOIRAI_CTX_LEN   = 200
MOIRAI_PRED_LEN  = 10
MOIRAI_BATCH     = 20
TOP_K_COVARIATES = 10   # how many MOIRAI-ranked symbols to feed Chronos

# Chronos
CHRONOS_MODEL_ID = "amazon/chronos-2"
CONTEXT_LENGTH   = 512

# WFO
WFO_TRAIN_MONTHS = 24
WFO_TEST_MONTHS  = 6
WFO_PURGE_DAYS   = 5
OOS_TEST_START   = "2018-04-01"   # confirm vs PLATTS history once loaded
LORA_BASE_SEED   = 42

# Sizing
TARGET_DAILY_VOL  = 0.015     # raw futures, not hedged equity
VOL_FLOOR         = 0.005
VOL_CAP           = 0.04
DET_SIGNAL_MAG    = 0.02      # ~2% — RB daily return scale
TRANSACTION_COST_BPS = 1.0    # CME-cheap; sweep [1, 5, 10]

# Signal
SMA_WINDOW       = 10
CONFIRM_DAYS     = 2
```

---

## 10. Scripts (entry points)

| # | Script | Inputs | Outputs |
|---|---|---|---|
| 01 | `01_extract_platts_candidates.py` | env vars, `--products`, `--start` | `outputs/<run>/data/platts_candidates.parquet` |
| 02 | `02_run_moirai_discovery.py` | `--run-dir`, `--anchor RB` | `outputs/<run>/data/moirai_ranking.csv` |
| 03 | `03_build_master_dataset.py` | `--run-dir`, `--product RB`, `--top-k 10` | `outputs/<run>/data/master.parquet` |
| 04 | `04_run_finetune_walkforward.py` | `--run-dir` | `outputs/<run>/predictions/fold_NN.parquet` |
| 05 | `05_run_ab_backtest.py` | `--run-dir`, `--bps 1 5 10` | `outputs/<run>/backtest/results.csv` |

All five accept `--run-dir`; if omitted, latest dir is auto-selected (like refiner).

---

## 11. Tests (lightweight, run with `pytest`)

| Test | What it checks |
|---|---|
| `test_snowflake_filter_sql.py` | SQL renders correctly for a sample whitelist; no SQL injection from product list |
| `test_futures_loader.py` | Log returns equal `ln(S_t / S_{t-1})`; no peek; fixed-tenor switches on expiry, not on volume |
| `test_sizing.py` | Each of 6 schemes returns 0 at p_up=0.5; vol cap kicks in at the right RV; DET sign matches signal sign |
| `test_walkforward_boundaries.py` | Train end is exactly `test_start - 6 days`; fold seeds differ across folds |

---

## 12. Risks, unknowns, and explicit deferrals

### 12.1 Known unknowns we accept now

| Unknown | Resolution |
|---|---|
| Exact PRODUCT names in `CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY` | `SELECT DISTINCT PRODUCT` before finalizing the whitelist (10-second query) |
| train.zip schema | Inspect once downloaded; loader is small enough to adapt |
| MOIRAI rank stability across re-runs | Validate by running discovery on two non-overlapping date halves and checking rank correlation |
| Right `TOP_K_COVARIATES` | Sweep [5, 10, 20]; pick by OOS Sharpe |

### 12.2 Risks

| Risk | Mitigation |
|---|---|
| MOIRAI's "global influence" is noisy on Z-scores | Cross-check with a simple linear feature-importance baseline (LASSO on RB return ~ Z-scores) |
| Futures returns are auto-correlated; Chronos may just learn lag-1 | Add a "naive lag-1" benchmark alongside SPY-style buy-and-hold |
| Pre-filter PRODUCT list misses a high-value driver | After first run, widen the whitelist and rerun MOIRAI; compare top-K stability |
| 8GB VRAM limits MOIRAI batch size | Pre-filter aggressively (§3.2); current code already paginates at BATCH_SIZE=20 |
| Roll discontinuities in train.zip | Use fixed-tenor (default 3rd prompt) per refiner; fall back to front-month only on insufficient coverage |

### 12.3 Out of scope for v0.1

- Spreads (RBCL, RBHO) — single-name baseline must land first
- Intraday signals — daily only
- Multi-product simultaneous training — RB, HO, CL run as separate experiments
- Live trading / paper trading — backtest only
- Survivorship-bias audit of PLATTS symbols — assume the whitelist is alive across the window

---

## 13. Definition of done for v0.1

1. `scripts/01..05` all run end-to-end on the user's machine after `train.zip` is unpacked.
2. `outputs/<run>/backtest/results.csv` contains, for **RB**, all six sizing schemes with Sharpe, annualized return, max DD, hit rate, num_trades, and txn cost at 1, 5, 10 bps RT.
3. A PnL chart compares each scheme to RB buy-and-hold and cash.
4. At least one scheme either:
   - Beats RB buy-and-hold by ≥ 1pp annualized at 5 bps, OR
   - Has Sharpe ≥ 0.4 in a regime where RB B&H Sharpe is ≤ 0.2 (defensive value).
5. Tests pass.
6. `docs/walkthrough.md` is written.

If (4) fails on RB, repeat steps 01–05 for HO and CL before reconsidering the framework.

---

## 14. Decisions still open — please confirm before code starts

1. **Folder name.** `platts/` (mirrors `commodity/` — named after input data). Alternative: `futures/` (mirrors `refiner/` — named after the trade target). Default: keep `platts/`.
2. **First target.** Confirm RB. (Stated in our chat, capturing it here.)
3. **MOIRAI selector option A vs B** (§3.3). Default: A (single-pass with RB in the matrix).
4. **DET signal flavor** (§6). Default: 6.1 (top-driver SMA crossover).
5. **Run discovery once or per-fold?** Default: **once on the full pre-OOS window** to avoid letting MOIRAI ranks drift across folds. (Stricter purist alternative: re-rank inside each fold. Defer to v0.2.)

If you OK these, next step is to scaffold the package (`platts/platts_strategy/...`) with empty stubs and the `config.py` filled in, then implement script 01 first.
