"""Sweep SPY-default overlay configurations against SPY baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import latest_run_dir
from refiner_strategy.evaluation.ab_runner import replay_with_predictions
from refiner_strategy.evaluation.spy_default_simulator import (
    SpyDefaultConfig,
    compare_to_spy_only,
    fetch_unhedged_returns,
    simulate_spy_default,
)
from refiner_strategy.finetune.walkforward import load_all_predictions
from refiner_strategy.signals.det_signal import build_stitched_det_signal
from refiner_strategy.data.build_dataset import build_master_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY-default simulation sweep")
    parser.add_argument("--bps", nargs="+", type=float, default=[10, 15, 20, 25])
    parser.add_argument("--borrow", nargs="+", type=float, default=[0, 50])
    parser.add_argument("--schemes", nargs="+", default=["DET", "ENS_VETO", "ENS_AVG", "NEW_CAP"])
    parser.add_argument("--run-dir", type=str, default=None)
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

    test_end = str(master.index.max().date())
    det_sig = build_stitched_det_signal(test_end=test_end)
    det_lagged = det_sig.shift(1).fillna(0)

    # Run A/B to get trades
    ab_results = replay_with_predictions(master, preds, det_lagged, schemes=tuple(args.schemes))

    # Get SPY returns and unhedged refiner returns
    spy_ret = master["SPY_Return"]
    refiner_ret = fetch_unhedged_returns(test_end=test_end)

    # SPY baseline
    spy_baseline = compare_to_spy_only(spy_ret)
    print(f"SPY buy-and-hold: Ann Ret={spy_baseline['ann_ret']:+.2%}, Sharpe={spy_baseline['sharpe']:.2f}")

    # Sweep
    rows = []
    for scheme in args.schemes:
        trades_df = ab_results[scheme]["trades"]
        if trades_df.empty:
            continue

        for bps in args.bps:
            for borrow in args.borrow:
                config = SpyDefaultConfig(
                    txn_cost_bps_per_leg=bps / 2,
                    borrow_cost_bps_per_year=borrow,
                )
                result = simulate_spy_default(
                    spy_returns=spy_ret,
                    refiner_returns=refiner_ret,
                    trades_df=trades_df,
                    config=config,
                )
                m = result["metrics"]
                edge = m.get("ann_ret", 0) - spy_baseline["ann_ret"]
                rows.append({
                    "scheme": scheme,
                    "txn_cost_bps_rt": bps,
                    "borrow_bps": borrow,
                    "ann_ret": m.get("ann_ret", 0),
                    "sharpe": m.get("sharpe", 0),
                    "max_dd_pct": m.get("max_dd_pct", 0),
                    "edge_vs_spy_pp": edge * 100,
                })

    results_df = pd.DataFrame(rows)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "spy_default_simulation.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n{'Scheme':<12} {'BPS RT':>8} {'Borrow':>8} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'Edge':>8}")
    print("-" * 68)
    for _, r in results_df.iterrows():
        print(
            f"{r['scheme']:<12} {r['txn_cost_bps_rt']:>8.0f} {r['borrow_bps']:>8.0f} "
            f"{r['ann_ret']:>+10.2%} {r['sharpe']:>8.2f} {r['max_dd_pct']:>8.1%} {r['edge_vs_spy_pp']:>+8.2f}pp"
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
