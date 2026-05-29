# Platts → CME Futures Strategy

Forecast and trade CME energy futures (RB, HO, CL) on a daily horizon using PLATTS physical-market features selected by MOIRAI and forecast by Chronos-2.

## Module relationship

```
commodity/  →  PLATTS data engineering (Snowflake ELT, normalization, Z-scores)
refiner/    →  Refiner-equity trading strategy (Chronos-2 + crack-spread DET)
platts/     →  CME-futures trading strategy (Chronos-2 + MOIRAI-selected PLATTS DET)
```

`platts/` reuses:
- `commodity/utility/snowflake_client.py` to pull PLATTS Z-scores
- `commodity/ml/inference.py` (MOIRAI attention sweep) for feature discovery
- `refiner/refiner_strategy/refiner_strategy/finetune/walkforward.py` (purged WFO) as the Chronos-2 fine-tune template
- `refiner/refiner_strategy/refiner_strategy/evaluation/ab_runner.py` (A/B harness) as the backtest template

## Status

**Planning phase.** See `docs/Implementation_Plan.md` for the full plan. No code has been written yet.

## Y target

CME futures daily returns, one product at a time. First: **RB** (RBOB gasoline). Then HO, CL. Spreads (e.g. RBCL) come after the single-name baseline lands.

## X target

PLATTS Z-scores selected by MOIRAI's attention-based Global Influence Score, anchored on the Y future of interest. Top-K (≈5–20) covariates are fed to Chronos-2 alongside the target.

## Execution timing

Futures settle at **2:30 PM ET**. The plan models a same-day "settle-to-settle" cycle: use Day X's settlement (and prior) to size a position held overnight, realized at Day X+1's settle. No equity trading; the asset *is* the future.
