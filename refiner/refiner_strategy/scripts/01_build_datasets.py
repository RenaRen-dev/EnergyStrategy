"""Build master dataset from yfinance and save to outputs directory."""
from __future__ import annotations

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
