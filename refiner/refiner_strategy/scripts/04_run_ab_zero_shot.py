"""Run A/B evaluation with live zero-shot Chronos inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import OOS_TEST_START, latest_run_dir
from refiner_strategy.evaluation.ab_runner import run_ab_zero_shot
from refiner_strategy.signals.det_signal import build_stitched_det_signal
from refiner_strategy.sizing.schemes import DEFAULT_SCHEMES
from refiner_strategy.utils.torch_helpers import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B zero-shot evaluation")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
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
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)

    det_sig = build_stitched_det_signal(test_end=args.end)
    det_lagged = det_sig.shift(1).fillna(0)

    # Load Chronos pipeline
    device = select_device()
    print(f"Device: {device}")

    try:
        import torch
        from chronos import Chronos2Pipeline
        from refiner_strategy.config import CHRONOS_MODEL_ID

        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL_ID,
            device_map=device,
            dtype=torch.float32,
        )
    except Exception as e:
        print(f"Could not load Chronos pipeline: {e}")
        print("Running with neutral predictions (no AI signal)")
        pipeline = None

    print(f"Running zero-shot A/B from {args.start} to {args.end or 'latest'}...")
    results = run_ab_zero_shot(
        master,
        start_date=args.start,
        end_date=args.end or str(master.index.max().date()),
        det_lagged=det_lagged,
        pipeline=pipeline,
    )

    print(f"\n{'Scheme':<12} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'Hit Rate':>10} {'N':>6}")
    print("-" * 58)
    for scheme in DEFAULT_SCHEMES:
        m = results[scheme]
        print(
            f"{scheme:<12} {m['ann_ret']:>+10.2%} {m['sharpe']:>8.2f} "
            f"{m['max_dd_pct']:>8.1%} {m['hit_rate']:>10.3f} {m['n']:>6}"
        )


if __name__ == "__main__":
    main()
