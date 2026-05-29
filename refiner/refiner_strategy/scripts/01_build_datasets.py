"""Build master dataset from yfinance and save to outputs directory."""
#  The output starts at 2016-01-11 because the Z-score computation requires a 256-day warmup window from __future__ import annotations
# So the first 256 trading days (roughly 1 year of 2015) produce NaN for Crack_Z_Score — those rows are dropped. That's why the dataset starts ~2016-01-11 instead of 2015-01-01.
# $VENV scripts/01_build_datasets.py --label m3_real --end 2026-05-27

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from refiner_strategy.config import OUTPUTS_DIR
from refiner_strategy.data.build_dataset import build_master_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master dataset from yfinance")
    parser.add_argument("--label", type=str, default="default", help="Run label")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"{timestamp}_{args.label}"
    dataset_dir = run_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building master dataset (end={args.end})...")
    df = build_master_dataset(end=args.end)
    out_path = dataset_dir / "master.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
