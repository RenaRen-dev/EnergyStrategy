"""Inspect existing prediction data to diagnose all-zero results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

pred_dir = Path("refiner_strategy/outputs/20260521_221630_myrun/predictions")

# Load all folds
all_preds = []
for i in range(17):
    f = pd.read_parquet(pred_dir / f"fold_{i:02d}.parquet")
    all_preds.append(f)
    nz = (f["q50"] != 0).sum()
    print(f"fold_{i:02d}: {len(f)} rows, non-zero q50: {nz}, NaN q50: {f['q50'].isna().sum()}")

combined = pd.concat(all_preds, ignore_index=True)
print(f"\nCombined: {len(combined)} predictions")
print(f"Columns: {list(combined.columns)}")
print(f"\nq50 stats:")
print(combined["q50"].describe())
print(f"\nq10 stats:")
print(combined["q10"].describe())
print(f"\nq90 stats:")
print(combined["q90"].describe())
print(f"\np_up stats:")
print(combined["p_up"].describe())

# Check if all predictions are effectively NEUTRAL
all_zero_q50 = (combined["q50"] == 0).all()
all_half_pup = (combined["p_up"] == 0.5).all()
print(f"\nAll q50 == 0? {all_zero_q50}")
print(f"All p_up == 0.5? {all_half_pup}")

# Sample some non-zero predictions if they exist
nz = combined[combined["q50"] != 0]
if len(nz) > 0:
    print(f"\n{len(nz)} non-zero q50 predictions. Sample:")
    print(nz.head(5).to_string())
else:
    print("\nALL predictions have q50 == 0!")

# Check the forecast vol gate
from refiner_strategy.config import Q90_Q10_TO_SIGMA, VOL_FLOOR
combined["forecast_vol"] = (combined["q90"] - combined["q10"]) / Q90_Q10_TO_SIGMA
above_floor = (combined["forecast_vol"] >= VOL_FLOOR).sum()
print(f"\nForecast vol >= VOL_FLOOR ({VOL_FLOOR}): {above_floor} / {len(combined)}")
print(f"forecast_vol stats:")
print(combined["forecast_vol"].describe())

# Check consensus gate
consensus = ((combined["q50"] > 0) == (combined["p_up"] > 0.5))
print(f"\nConsensus agreement: {consensus.sum()} / {len(combined)}")
