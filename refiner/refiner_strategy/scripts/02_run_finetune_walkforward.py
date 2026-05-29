"""Run 17-fold LoRA walk-forward fine-tuning on Chronos-2."""

'''
  The training window is a rolling fixed 24 months — it does not grow.
  Look at the folds:  
  fold 00: train 2016-03-26 .. 2018-03-26  (24 mo)
  fold 16: train 2024-03-26 .. 2026-03-26  (24 mo)
  Each fold trains on exactly 24 months before its test window. So even with 10 years of data, every fold still only trains on its own rolling 2-year
  slice. Adding history does not make any fold train on more data.

  This is all governed by config:
  OOS_TEST_START = "2018-04-01"   # where folds begin (fixed)
  WFO_TRAIN_MONTHS = 24           # rolling train window (fixed size)
  WFO_TEST_MONTHS  = 6            # test window size → fold count
'''



'''
5 folds (was 17):
    fold 00: train 2016-12 .. 2021-12 (60mo) | test 2022   ← README split exactly
    fold 01: train 2017-12 .. 2022-12 (60mo) | test 2023
    fold 02: train 2018-12 .. 2023-12 (60mo) | test 2024
    fold 03: train 2019-12 .. 2024-12 (60mo) | test 2025
    fold 04: train 2020-12 .. 2025-12 (60mo) | test 2026 (partial → 05-26)
'''



from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import latest_run_dir
from refiner_strategy.finetune.walkforward import run_all_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward fine-tuning")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (default: latest)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run_dir()
        if run_dir is None:
            print("No run directory found. Run 01_build_datasets.py first.")
            sys.exit(1)

    master_path = run_dir / "datasets" / "master.csv"
    if not master_path.exists():
        print(f"Master dataset not found at {master_path}")
        sys.exit(1)

    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running walk-forward fine-tuning from {run_dir}")
    print(f"Master dataset: {len(master)} rows")

    preds = run_all_folds(master, pred_dir)
    print(f"Generated {len(preds)} predictions across all folds")
    print(f"Predictions saved to {pred_dir}")


if __name__ == "__main__":
    main()
