"""Long/short simulation: refiner book (long + short) vs buy-and-hold benchmarks.

Companion to ``05_run_spy_default_simulation.py``.  Where 05 parks unused
capital in SPY and only goes long, this script runs a true **long/short** book:
the DET (M6 crack) and trained-Chronos schemes go +1 long / −1 short on the
beta-hedged refiner names.  Results are reported over the README test window
(trade_date ≥ TEST_START) against three benchmarks:

  - each underlying stock buy-and-hold (single),
  - the cap-weighted B7 basket buy-and-hold (combined with percentage),
  - SPY buy-and-hold.

Uses the already-trained model: predictions are replayed from the run's stored
``fold_*.parquet`` files (no Chronos inference here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import TEST_START, latest_run_dir
from refiner_strategy.evaluation.ab_runner import replay_with_predictions
from refiner_strategy.evaluation.benchmarks import (
    basket_benchmark,
    single_stock_benchmarks,
    spy_benchmark,
)
from refiner_strategy.evaluation.long_short_simulator import (
    LongShortConfig,
    simulate_long_short,
)
from refiner_strategy.evaluation.spy_default_simulator import fetch_per_ticker_returns
from refiner_strategy.finetune.walkforward import load_all_predictions
from refiner_strategy.signals.det_signal import build_stitched_det_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Long/short simulation sweep")
    # Cost SCENARIOS are paired element-wise (zip), not crossed.  The README
    # specifies no costs, so the default primary is cost-free (0bps / 0 borrow),
    # with one light sensitivity scenario (10bps round-trip / 50bps-yr borrow).
    parser.add_argument("--bps", nargs="+", type=float, default=[0, 10],
                        help="Round-trip transaction costs (bps), paired with --borrow")
    parser.add_argument("--borrow", nargs="+", type=float, default=[0, 50],
                        help="Short borrow costs (bps/yr), paired with --bps")
    parser.add_argument("--schemes", nargs="+", default=["DET", "NEW_CAP", "ENS_VETO", "ENS_AVG"])
    parser.add_argument("--test-start", type=str, default=TEST_START, help="Evaluation start (README test window)")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run_dir()
        if run_dir is None:
            print("No run directory found. Run 01_build_datasets.py first.")
            sys.exit(1)

    if len(args.bps) != len(args.borrow):
        print(f"--bps ({len(args.bps)}) and --borrow ({len(args.borrow)}) must have "
              f"equal length (they are paired element-wise, not crossed).")
        sys.exit(1)
    cost_scenarios = list(zip(args.bps, args.borrow))

    master_path = run_dir / "datasets" / "master.csv"
    pred_dir = run_dir / "predictions"

    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    preds = load_all_predictions(pred_dir)

    test_end = str(master.index.max().date())
    det_sig = build_stitched_det_signal(test_end=test_end)
    det_lagged = det_sig.shift(1).fillna(0)

    # Replay trained-model + DET schemes to get the per-ticker trade log.
    ab_results = replay_with_predictions(master, preds, det_lagged, schemes=tuple(args.schemes))

    test_start = pd.Timestamp(args.test_start)

    # ── Benchmarks (buy-and-hold over the test window) ──
    spy_ret = master["SPY_Return"]
    ticker_rets = fetch_per_ticker_returns(test_end=test_end)

    spy_bm = spy_benchmark(spy_ret, start=args.test_start)
    basket_bm = basket_benchmark(ticker_rets, start=args.test_start)
    single_bm = single_stock_benchmarks(ticker_rets, start=args.test_start)

    # Helpers for the README-style columns ($ PnL in $M, positive-year count).
    def m_pnl(m: dict) -> str:
        return f"{m.get('total_pnl', 0.0) / 1e6:>+8.2f}M"

    def m_years(m: dict) -> str:
        return f"{m.get('n_pos_years', 0)}/{m.get('n_years', 0)}"

    print(f"\n=== Buy-and-hold benchmarks (≥ {args.test_start}, $10M notional) ===")
    hdr = f"{'Benchmark':<12} {'TotRet':>9} {'Ann Ret':>9} {'Sharpe':>7} {'MaxDD':>7} {'Tot $PnL':>10} {'PosYrs':>7}"
    print(hdr); print("-" * len(hdr))

    def print_bm(name: str, m: dict) -> None:
        print(f"{name:<12} {m['total_return']:>+9.1%} {m['ann_ret']:>+9.2%} {m['sharpe']:>7.2f} "
              f"{m['max_dd_pct']:>7.1%} {m_pnl(m):>10} {m_years(m):>7}")

    print_bm("SPY", spy_bm)
    print_bm("B7 basket", basket_bm)
    for t, m in single_bm.items():
        print_bm(t, m)

    # ── Strategy sweep ──
    rows = []
    primary_scenario = cost_scenarios[0]  # per-year table uses the first (primary) scenario
    yearly_pnl: dict[str, pd.Series] = {}  # scheme -> per-year $ PnL (primary scenario)
    for scheme in args.schemes:
        trades_df = ab_results[scheme]["trades"]
        if trades_df.empty:
            continue
        trades_df = trades_df[pd.to_datetime(trades_df["date"]) >= test_start]
        if trades_df.empty:
            continue

        for bps, borrow in cost_scenarios:
            label = "cost-free" if (bps == 0 and borrow == 0) else f"{bps:.0f}bps/{borrow:.0f}bp"
            config = LongShortConfig(
                txn_cost_bps_per_leg=bps / 2,
                borrow_cost_bps_per_year=borrow,
            )
            result = simulate_long_short(trades_df, config=config)
            m = result["metrics"]
            if (bps, borrow) == primary_scenario:
                yearly_pnl[scheme] = m.get("pnl_by_year", pd.Series(dtype=float))
            rows.append({
                "scheme": scheme,
                "scenario": label,
                "txn_cost_bps_rt": bps,
                "borrow_bps": borrow,
                "ann_ret": m.get("ann_ret", 0),
                "total_return": m.get("total_return", 0),
                "sharpe": m.get("sharpe", 0),
                "sharpe_invested": m.get("sharpe_invested", 0),
                "max_dd_pct": m.get("max_dd_pct", 0),
                "total_pnl": m.get("total_pnl", 0),
                "n_pos_years": m.get("n_pos_years", 0),
                "n_years": m.get("n_years", 0),
                "edge_vs_spy_pp": (m.get("ann_ret", 0) - spy_bm["ann_ret"]) * 100,
            })

    results_df = pd.DataFrame(rows)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "long_short_simulation.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n=== Long/short strategy ({len(cost_scenarios)} cost scenario(s), ≥ {args.test_start}, $10M notional) ===")
    hdr = (f"{'Scheme':<10} {'Scenario':>10} {'Ann Ret':>9} {'Sharpe':>7} {'Shrp(inv)':>9} "
           f"{'MaxDD':>7} {'Tot $PnL':>10} {'PosYrs':>7} {'Edge':>9}")
    print(hdr); print("-" * len(hdr))
    for _, r in results_df.iterrows():
        print(
            f"{r['scheme']:<10} {r['scenario']:>10} {r['ann_ret']:>+9.2%} {r['sharpe']:>7.2f} "
            f"{r['sharpe_invested']:>9.2f} {r['max_dd_pct']:>7.1%} {r['total_pnl']/1e6:>+8.2f}M "
            f"{int(r['n_pos_years'])}/{int(r['n_years']):>2} {r['edge_vs_spy_pp']:>+7.2f}pp"
        )

    # ── Per-year P&L breakdown ("P&L in each year"), primary scenario ──
    prim_label = "cost-free" if primary_scenario == (0, 0) else f"{primary_scenario[0]:.0f}bps/{primary_scenario[1]:.0f}bp"
    year_cols = {"SPY": spy_bm["pnl_by_year"], "B7": basket_bm["pnl_by_year"], **yearly_pnl}
    year_table = pd.DataFrame(year_cols).sort_index()
    if not year_table.empty:
        print(f"\n=== Per-year P&L ($M, {prim_label} scenario) ===")
        cols = list(year_table.columns)
        print(f"{'Year':<6}" + "".join(f"{c:>10}" for c in cols))
        print("-" * (6 + 10 * len(cols)))
        for yr, row in year_table.iterrows():
            print(f"{int(yr):<6}" + "".join(f"{(row[c]/1e6):>+10.2f}" for c in cols))
        # totals + positive-year counts per column
        print("-" * (6 + 10 * len(cols)))
        print(f"{'Total':<6}" + "".join(f"{(year_table[c].sum()/1e6):>+10.2f}" for c in cols))
        print(f"{'+Yrs':<6}" + "".join(
            f"{f'{int((year_table[c] > 0).sum())}/{int(year_table[c].notna().sum())}':>10}" for c in cols
        ))
        year_table.to_csv(results_dir / "long_short_pnl_by_year.csv", index_label="year")

    print(f"\nSaved to {out_path}")
    print(f"Saved per-year P&L to {results_dir / 'long_short_pnl_by_year.csv'}")


if __name__ == "__main__":
    main()
