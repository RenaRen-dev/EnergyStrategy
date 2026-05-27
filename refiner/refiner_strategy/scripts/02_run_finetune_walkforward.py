"""Run 17-fold LoRA walk-forward fine-tuning on Chronos-2."""
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
