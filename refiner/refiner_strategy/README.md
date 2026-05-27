# Refiner Strategy

Long/short equity strategy on 7 US oil refiner stocks, driven by crack-spread signals and Chronos-2 AI forecasts.

## What This Strategy Is

US oil refiners buy crude and sell gasoline and heating oil. Their profit margin -- the "crack spread" -- is an accounting identity: it equals the sale price of refined products minus the cost of crude. When the crack spread widens, refiners earn more per barrel; when it narrows, margins compress. This strategy trades refiner equities based on the direction of the crack spread, hedged against the broad market via rolling beta.

The thesis is simple: if you can predict the direction of the crack spread one day ahead, you can predict which way refiner stocks will move, because the spread **is** their margin.

## Asset Universe

| Ticker | Name | Weight |
|--------|------|--------|
| VLO | Valero Energy | 25% |
| MPC | Marathon Petroleum | 25% |
| PSX | Phillips 66 | 25% |
| DINO | HF Sinclair | 10% |
| PBF | PBF Energy | 5% |
| DK | Delek US Holdings | 5% |
| CVI | CVR Energy | 5% |

All price data is sourced from yfinance at runtime. No bundled CSV data files.

## Data Flow

1. **Download** -- Futures (CL, RB, HO) and equity prices from yfinance
2. **Compute** -- 3:2:1 crack spread, rolling Z-score, beta-hedged returns
3. **Signal** -- Deterministic SMA crossover (DET) and/or Chronos-2 quantile forecasts
4. **Size** -- Six sizing schemes convert signals into dollar positions
5. **Evaluate** -- A/B harness computes PnL, Sharpe, drawdown, hit rate

## Two Forecast Signals

**DET (Deterministic):** The Houston Products Desk's rule-based signal. Computes a 10-day SMA of the 3:2:1 crack spread. When the crack is above the SMA for 2+ consecutive days, go long refiners; below for 2+ days, go short. Applied with a T+1 lag (today's signal uses yesterday's close).

**Chronos-2 (AI):** Amazon's foundation time-series model, fine-tuned with LoRA adapters in a walk-forward loop. Produces quantile forecasts (q10 through q90) of next-day hedged returns. Used as a defensive filter or combined with DET via ensemble methods.

## Six Sizing Schemes

| Scheme | Description | Reference |
|--------|-------------|-----------|
| OLD | Probability-only with deadband (the buggy original) | -- |
| NEW | Vol-targeted with consensus gate | Moskowitz-Ooi-Pedersen 2012 |
| NEW_CAP | NEW + realised-vol cap | Pedersen 2015 |
| DET | Pure deterministic crack-spread signal | -- |
| ENS_VETO | Both signals must agree on direction | -- |
| ENS_AVG | Bates-Granger forecast combination of Chronos and DET | Bates-Granger 1969 |

## Walk-Forward Methodology

- 17 non-overlapping folds covering 2018-04 to 2026-05 (8.1 years OOS, 2044 trading days)
- Fixed 24-month trailing training window per fold
- 6-month test window per fold
- 5-day purge gap between train and test (no information leakage)
- Per-fold deterministic seeding: `fold_seed = LORA_BASE_SEED + fold_idx`

## A/B Harness

The evaluation harness runs all 6 sizing schemes in parallel over the same date range, using identical predictions and market data. This enables pairwise comparison: every difference in PnL between two schemes is attributable solely to the sizing logic, not to data differences.

## Four Bugs Found

| ID | Bug | Impact |
|----|-----|--------|
| H1 | Z-score computed per-slice instead of unified | Discontinuity at fold boundaries |
| H3 | Same random seed for all folds | Non-reproducible fine-tuning |
| H4 | Hit rate used target_size instead of effective_size | Overstated accuracy by ~3pp |
| H5 | Rolling beta used today's return | Look-ahead bias in hedge ratio |

## Final Results

8.1-year OOS backtest (2018-04 to 2026-05, 2044 trading days, 10 bps RT transaction cost):

| Scheme | Ann Return | Sharpe | Max DD |
|--------|-----------|--------|--------|
| OLD | -4.15% | -0.29 | -- |
| NEW | -1.97% | +0.14 | -- |
| NEW_CAP | +2.87% | +0.24 | -- |
| DET | +8.87% | +0.45 | -60% |
| ENS_VETO | +7.78% | +0.51 | -37% |
| ENS_AVG | +8.97% | +0.50 | -44% |
| SPY (benchmark) | +13.94% | +0.82 | -34% |

## SPY-Default Overlay

The more realistic question: "What happens if unused capital sits in SPY instead of cash?"

Best configuration: **ENS_VETO @ 10 bps RT = +16.37% ann, Sharpe +0.74, +2.36pp edge vs SPY buy-and-hold.**

This is the only configuration that beats SPY as a standalone investment.

## Alpha Search Ledger

Nine additional strategies were tested and rejected:

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

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python scripts/01_build_datasets.py --label myrun
python scripts/02_run_finetune_walkforward.py
python scripts/03_run_ab_finetuned.py
python scripts/04_run_ab_zero_shot.py --start 2022-01-01
python scripts/05_run_spy_default_simulation.py --bps 10 15 20 25
```

## Repository Layout

```
refiner_strategy/
  refiner_strategy/           # importable package
    config.py                 # all knobs -- single source of truth
    data/
      build_dataset.py        # master dataset builder from yfinance
      futures_loader.py       # continuous crack spread loader
    signals/
      det_signal.py           # deterministic crack-spread SMA
    sizing/
      schemes.py              # 6 sizing functions + SIZERS dict
    evaluation/
      metrics.py              # Sharpe, drawdown, hit rate
      ab_runner.py            # live + replay A/B harness
      spy_default_simulator.py
    finetune/
      walkforward.py          # 17-fold LoRA loop
    utils/
      torch_helpers.py        # device selection
  scripts/                    # 5 entry points
  docs/
    walkthrough.md            # plain-language walkthrough
  tests/
    test_sizing.py            # 9 unit tests
  outputs/                    # created at runtime
```

## Honest Limitations

1. **Does not beat SPY standalone.** The best standalone scheme (ENS_AVG, +8.97%) trails SPY (+13.94%) by 5pp annually. Only the SPY-default overlay with ENS_VETO at optimistic 10 bps costs edges SPY.

2. **Roll noise in futures.** CL=F, RB=F, HO=F are front-month continuous contracts with roll gaps. The crack spread inherits this noise. No back-adjustment is applied.

3. **Optimistic transaction costs.** 10 bps round-trip is aggressive for small-cap refiners like CVI and DK. Real costs may be 2-5x higher for those names.

4. **No slippage model.** Positions are filled at the close price with no market impact.

5. **Survivorship bias.** The 7-stock universe was selected with hindsight. Stocks that delisted or were acquired during the backtest period are excluded.

6. **Single-asset-class concentration.** The strategy is 100% exposed to the energy sector. Sector drawdowns (e.g., 2020 COVID crash) hit all positions simultaneously.

7. **Chronos model risk.** The AI component depends on a specific pre-trained model (amazon/chronos-2) that may be updated, deprecated, or behave differently on new data.

8. **No live trading validation.** All results are from backtests. Paper trading and live execution may reveal additional issues.
