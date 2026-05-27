"""Run A/B evaluation using precomputed walk-forward predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import latest_run_dir
from refiner_strategy.evaluation.ab_runner import replay_with_predictions
from refiner_strategy.finetune.walkforward import load_all_predictions
from refiner_strategy.signals.det_signal import build_stitched_det_signal
from refiner_strategy.sizing.schemes import DEFAULT_SCHEMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B evaluation with finetuned predictions")
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
    pred_dir = run_dir / "predictions"

    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    preds = load_all_predictions(pred_dir)

    # Build DET signal and lag by 1
    test_end = str(master.index.max().date())
    det_sig = build_stitched_det_signal(test_end=test_end)
    det_lagged = det_sig.shift(1).fillna(0)

    print("Running A/B evaluation (finetuned)...")
    results = replay_with_predictions(master, preds, det_lagged)

    # Print results table
    print(f"\n{'Scheme':<12} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'Hit Rate':>10} {'N':>6}")
    print("-" * 58)
    for scheme in DEFAULT_SCHEMES:
        m = results[scheme]
        print(
            f"{scheme:<12} {m['ann_ret']:>+10.2%} {m['sharpe']:>8.2f} "
            f"{m['max_dd_pct']:>8.1%} {m['hit_rate']:>10.3f} {m['n']:>6}"
        )

    # Save results
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scheme in DEFAULT_SCHEMES:
        m = results[scheme]
        rows.append({"scheme": scheme, **{k: v for k, v in m.items() if k not in ("daily_pnl", "trades")}})
    pd.DataFrame(rows).to_csv(results_dir / "ab_finetuned_pooled.csv", index=False)
    print(f"\nResults saved to {results_dir / 'ab_finetuned_pooled.csv'}")


if __name__ == "__main__":
    main()
