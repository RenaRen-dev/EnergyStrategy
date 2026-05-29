"""SPY-default overlay simulation: idle capital earns SPY returns.

Adapts refiner_strategy/evaluation/spy_default_simulator.py for single-asset RB.

Mechanics (per day t, single-asset version):
    target_rb_frac    = sizer_output / notional           (signed, can be < 0)
    target_rb_clipped = clip(target_rb_frac, -lev_cap, +lev_cap)
    target_spy_weight = 1.0 - |target_rb_clipped|         (residual capital -> SPY)

    gross_ret = prev_rb_frac * rb_simple_ret_t  +  prev_spy_weight * spy_simple_ret_t
    txn_cost  = (|d_rb_frac| + |d_spy_weight|) * bps_per_leg / 10000
    borrow    = max(0, -prev_rb_frac) * borrow_bps / 10000 / 252

    NAV_t = NAV_{t-1} * (1 + gross_ret - txn_cost - borrow)

PnL accounting uses YESTERDAY's positions (H4 invariant), then rebalances at close.

Inputs (read-only):
    platts/outputs/master_dataset.parquet
    platts/outputs/chronos_predictions/all_preds.parquet
    platts/outputs/platts_det_signal.parquet
    platts/data/train/SPY_daily.csv

Outputs:
    platts/outputs/backtest_spy_default/
        results.csv             # scheme x bps x borrow x metrics
        nav_<scheme>_<bps>bps_<borrow>bp_borrow.csv  # date x NAV
        summary.csv             # one-row-per-config table

Usage:
    python platts/scripts/09_run_spy_default_simulation.py
    python platts/scripts/09_run_spy_default_simulation.py --bps 5 --borrow 0 50
    python platts/scripts/09_run_spy_default_simulation.py --schemes ENS_VETO ENS_AVG
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_PATH = PROJECT_ROOT / "platts" / "outputs" / "master_dataset.parquet"
PRED_PATH   = PROJECT_ROOT / "platts" / "outputs" / "chronos_predictions" / "all_preds.parquet"
DET_PATH    = PROJECT_ROOT / "platts" / "outputs" / "platts_det_signal.parquet"
SPY_PATH    = PROJECT_ROOT / "platts" / "data" / "train" / "SPY_daily.csv"
OUT_DIR     = PROJECT_ROOT / "platts" / "outputs" / "backtest_spy_default"

# ---------------------------------------------------------------------------
# Constants (same as step 8)
# ---------------------------------------------------------------------------
TARGET_DAILY_VOL     = 0.015
VOL_FLOOR            = 0.005
VOL_CAP              = 0.04
DET_SIGNAL_MAG       = 0.02
Q90_Q10_TO_SIGMA     = 2.5631
OLD_DEADBAND_LOW     = 0.40
OLD_DEADBAND_HIGH    = 0.60
OLD_CONVICTION_SLOPE = 5.0
REALIZED_VOL_WIN     = 20
TRADING_DAYS         = 252

NEUTRAL_PRED = {
    "q10": 0.0, "q20": 0.0, "q30": 0.0, "q40": 0.0, "q50": 0.0,
    "q60": 0.0, "q70": 0.0, "q80": 0.0, "q90": 0.0, "p_up": 0.5,
}


# ===========================================================================
# Sizing schemes (copied from step 8 to keep this script self-contained)
# ===========================================================================
def _apply_vol_cap(base, rv):
    if rv is None or rv != rv or rv <= 0 or rv <= VOL_CAP:
        return base
    return base * (VOL_CAP / rv)

def size_old(pred, det_sig, notional, rv):
    p_up = pred.get("p_up", 0.5)
    if OLD_DEADBAND_LOW < p_up < OLD_DEADBAND_HIGH:
        return 0.0
    conviction = min(1.0, abs(p_up - 0.5) * OLD_CONVICTION_SLOPE)
    return (1.0 if p_up > 0.5 else -1.0) * conviction * notional

def size_new(pred, det_sig, notional, rv):
    q10, q50, q90 = pred.get("q10", 0.0), pred.get("q50", 0.0), pred.get("q90", 0.0)
    p_up = pred.get("p_up", 0.5)
    raw_fcst_vol = (q90 - q10) / Q90_Q10_TO_SIGMA
    if raw_fcst_vol < VOL_FLOOR or (q50 > 0) != (p_up > 0.5):
        return 0.0
    raw_size = (q50 / raw_fcst_vol) / TARGET_DAILY_VOL
    return max(-1.0, min(1.0, raw_size)) * notional

def size_new_cap(pred, det_sig, notional, rv):
    return _apply_vol_cap(size_new(pred, det_sig, notional, rv), rv)

def size_det(pred, det_sig, notional, rv):
    return 0.0 if det_sig == 0 else _apply_vol_cap(det_sig * notional, rv)

def size_ens_veto(pred, det_sig, notional, rv):
    base = size_new_cap(pred, det_sig, notional, rv)
    if base == 0.0 or det_sig == 0:
        return 0.0
    return base if (base > 0) == (det_sig > 0) else 0.0

def size_ens_avg(pred, det_sig, notional, rv):
    q10, q50, q90 = pred.get("q10", 0.0), pred.get("q50", 0.0), pred.get("q90", 0.0)
    p_up = pred.get("p_up", 0.5)
    raw_fcst_vol = (q90 - q10) / Q90_Q10_TO_SIGMA
    if raw_fcst_vol < VOL_FLOOR:
        return 0.0
    avg_q50 = 0.5 * (q50 + det_sig * DET_SIGNAL_MAG)
    if (avg_q50 > 0) != (p_up > 0.5):
        return 0.0
    raw_size = (avg_q50 / raw_fcst_vol) / TARGET_DAILY_VOL
    return _apply_vol_cap(max(-1.0, min(1.0, raw_size)) * notional, rv)

SIZERS = {
    "OLD": size_old, "NEW": size_new, "NEW_CAP": size_new_cap,
    "DET": size_det, "ENS_VETO": size_ens_veto, "ENS_AVG": size_ens_avg,
}


# ===========================================================================
# Data loaders
# ===========================================================================
def load_spy_simple_returns() -> pd.Series:
    """SPY adjusted-close simple daily returns (pct_change)."""
    spy = pd.read_csv(SPY_PATH, parse_dates=["date"]).set_index("date").sort_index()
    spy.index = spy.index.normalize()
    return spy["Close"].pct_change().dropna()


# ===========================================================================
# Single-asset SPY-default simulator
# ===========================================================================
def simulate(
    rb_simple_ret: pd.Series,
    spy_simple_ret: pd.Series,
    preds: pd.DataFrame,
    det_sig: pd.Series,
    sizer,
    notional: float,
    bps_rt: float,
    borrow_bps_yr: float,
    leverage_cap: float,
) -> dict:
    """One pass of the SPY-default sim for a single sizer."""
    bps_per_leg = bps_rt / 2.0   # round-trip split into entry + exit

    # Prediction lookup
    preds = preds.copy()
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.normalize()
    preds_idx = preds.set_index("Date")

    # DET lagged by 1 day (T+1 convention)
    det_lagged = det_sig.shift(1).fillna(0)

    # Realized vol of RB
    rv = rb_simple_ret.rolling(REALIZED_VOL_WIN, min_periods=5).std()

    # Common dates: preds ∩ rb ∩ spy
    dates = (preds_idx.index
             .intersection(rb_simple_ret.index)
             .intersection(spy_simple_ret.index)
             .intersection(det_lagged.index))
    dates = sorted(dates)

    nav = 1.0
    prev_rb_frac = 0.0
    prev_spy_weight = 1.0  # start fully in SPY
    nav_path = []
    total_txn = 0.0
    total_borrow = 0.0
    trades = 0
    active_days = 0
    hits = 0

    for T in dates:
        # --- Returns earned today (using YESTERDAY's allocation) ---
        rb_r  = float(rb_simple_ret.loc[T])
        spy_r = float(spy_simple_ret.loc[T])
        gross = prev_rb_frac * rb_r + prev_spy_weight * spy_r

        # --- Build today's target allocation ---
        if T in preds_idx.index:
            row = preds_idx.loc[T]
            pred = {k: float(row[k]) if k in row.index and row[k] == row[k] else NEUTRAL_PRED[k]
                    for k in NEUTRAL_PRED}
        else:
            pred = dict(NEUTRAL_PRED)
        det_today = float(det_lagged.loc[T])
        rv_today = rv.loc[T] if T in rv.index else None
        if rv_today is not None and rv_today != rv_today:
            rv_today = None

        target_dollar = sizer(pred, det_today, notional, rv_today)
        target_rb_frac = target_dollar / notional
        target_rb_frac = max(-leverage_cap, min(leverage_cap, target_rb_frac))
        target_spy_weight = 1.0 - abs(target_rb_frac)

        # --- Transaction costs on the rebalance ---
        rb_leg  = abs(target_rb_frac - prev_rb_frac) * (bps_per_leg / 10_000)
        spy_leg = abs(target_spy_weight - prev_spy_weight) * (bps_per_leg / 10_000)
        txn = rb_leg + spy_leg

        # --- Borrow cost on YESTERDAY's short position ---
        prev_short = max(0.0, -prev_rb_frac)
        borrow = prev_short * (borrow_bps_yr / 10_000) / TRADING_DAYS

        # --- NAV evolution ---
        nav *= (1.0 + gross - txn - borrow)
        nav_path.append((T, nav))
        total_txn += txn
        total_borrow += borrow

        # Stats
        if target_rb_frac != prev_rb_frac:
            trades += 1
        if prev_rb_frac != 0:
            active_days += 1
            if math.copysign(1, prev_rb_frac) == math.copysign(1, rb_r) and rb_r != 0:
                hits += 1

        # Roll state
        prev_rb_frac = target_rb_frac
        prev_spy_weight = target_spy_weight

    nav_series = pd.Series({d: v for d, v in nav_path}).sort_index()
    daily_ret = nav_series.pct_change().dropna()
    mu, sigma = daily_ret.mean(), daily_ret.std()
    sharpe = (math.sqrt(TRADING_DAYS) * mu / sigma) if sigma > 0 else 0.0
    ann_ret = mu * TRADING_DAYS
    ann_vol = sigma * math.sqrt(TRADING_DAYS)
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "nav_series":   nav_series,
        "ann_ret":      float(ann_ret),
        "ann_vol":      float(ann_vol),
        "sharpe":       float(sharpe),
        "max_dd":       max_dd,
        "final_nav":    float(nav_series.iloc[-1]),
        "txn_total":    total_txn,
        "borrow_total": total_borrow,
        "num_trades":   trades,
        "active_days":  active_days,
        "hit_rate":     hits / active_days if active_days > 0 else float("nan"),
    }


def spy_only_baseline(spy_simple_ret: pd.Series, dates) -> dict:
    """Pure SPY buy-and-hold on the same date set."""
    spy = spy_simple_ret.reindex(dates).fillna(0)
    nav = (1.0 + spy).cumprod()
    mu, sigma = spy.mean(), spy.std()
    sharpe = (math.sqrt(TRADING_DAYS) * mu / sigma) if sigma > 0 else 0.0
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    return {
        "nav_series": nav,
        "ann_ret":    float(mu * TRADING_DAYS),
        "ann_vol":    float(sigma * math.sqrt(TRADING_DAYS)),
        "sharpe":     float(sharpe),
        "max_dd":     float(drawdown.min()),
        "final_nav":  float(nav.iloc[-1]),
    }


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bps", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    p.add_argument("--borrow", type=float, nargs="+", default=[0.0, 50.0],
                   help="Short-side borrow cost in bps/year (default sweeps 0 and 50)")
    p.add_argument("--schemes", nargs="+", default=list(SIZERS.keys()))
    p.add_argument("--notional", type=float, default=100.0)
    p.add_argument("--leverage-cap", type=float, default=1.0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print(f"[LOAD] {MASTER_PATH.name}")
    master = pd.read_parquet(MASTER_PATH)
    master.index = pd.to_datetime(master.index).normalize()
    rb_log = master["RB_LogReturn"]
    rb_simple = np.expm1(rb_log)       # convert log -> simple

    print(f"[LOAD] {PRED_PATH.name}")
    preds = pd.read_parquet(PRED_PATH)

    print(f"[LOAD] {DET_PATH.name}")
    det = pd.read_parquet(DET_PATH)
    det.index = pd.to_datetime(det.index).normalize()
    det_sig = det["det_sig"]

    print(f"[LOAD] {SPY_PATH.name}")
    spy_simple = load_spy_simple_returns()
    print(f"       SPY rows: {len(spy_simple):,}")

    # Pre-compute common date set for the SPY baseline
    preds_dates = pd.to_datetime(preds["Date"]).dt.normalize().unique()
    common = (pd.DatetimeIndex(preds_dates)
              .intersection(rb_simple.index)
              .intersection(spy_simple.index)
              .intersection(det_sig.index))

    spy_base = spy_only_baseline(spy_simple, common)
    print(f"\n  SPY buy-and-hold baseline on {len(common)} days:")
    print(f"    ann_ret={spy_base['ann_ret']*100:+.2f}%  sharpe={spy_base['sharpe']:+.3f}  "
          f"max_dd={spy_base['max_dd']*100:+.2f}%  final_nav={spy_base['final_nav']:.3f}")

    # Save SPY baseline NAV
    spy_base["nav_series"].to_csv(OUT_DIR / "nav_SPY_baseline.csv", header=["NAV"])

    # Sweep
    rows = []
    for scheme in args.schemes:
        sizer = SIZERS[scheme]
        for bps in args.bps:
            for borrow in args.borrow:
                tag = f"{scheme}_{int(bps)}bps_{int(borrow)}bp_borrow"
                print(f"\n  Running {tag} ...")
                res = simulate(
                    rb_simple_ret=rb_simple,
                    spy_simple_ret=spy_simple,
                    preds=preds,
                    det_sig=det_sig,
                    sizer=sizer,
                    notional=args.notional,
                    bps_rt=bps,
                    borrow_bps_yr=borrow,
                    leverage_cap=args.leverage_cap,
                )
                edge = res["ann_ret"] - spy_base["ann_ret"]
                rows.append({
                    "scheme":     scheme,
                    "bps":        bps,
                    "borrow":     borrow,
                    "ann_ret":    res["ann_ret"],
                    "ann_vol":    res["ann_vol"],
                    "sharpe":     res["sharpe"],
                    "max_dd":     res["max_dd"],
                    "final_nav":  res["final_nav"],
                    "hit_rate":   res["hit_rate"],
                    "num_trades": res["num_trades"],
                    "txn_total":  res["txn_total"],
                    "borrow_total": res["borrow_total"],
                    "edge_vs_spy_pp": edge * 100,
                })
                # Save NAV path
                res["nav_series"].to_csv(OUT_DIR / f"nav_{tag}.csv", header=["NAV"])

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n[OK] summary -> {OUT_DIR / 'summary.csv'}")

    # Print table
    print("\n" + "=" * 100)
    print("  SPY-DEFAULT OVERLAY RESULTS  (unused capital earns SPY)")
    print("=" * 100)
    print(f"  {'scheme':<10} {'bps':>5} {'borrow':>7}  "
          f"{'ann_ret':>8} {'ann_vol':>7} {'sharpe':>7} {'max_dd':>8} "
          f"{'hit_rt':>7} {'trades':>7} {'edge':>10}")
    print("  " + "-" * 95)
    for _, r in summary.iterrows():
        hr = r["hit_rate"]
        hr_str = f"{hr*100:6.2f}%" if hr == hr else "   n/a"
        print(f"  {r['scheme']:<10} {r['bps']:>5.0f} {r['borrow']:>7.0f}  "
              f"{r['ann_ret']*100:>+7.2f}% {r['ann_vol']*100:>6.2f}% "
              f"{r['sharpe']:>+7.3f} {r['max_dd']*100:>+7.2f}% "
              f"{hr_str:>7} {r['num_trades']:>7d} {r['edge_vs_spy_pp']:>+8.2f}pp")
    print(f"\n  SPY baseline: ann={spy_base['ann_ret']*100:+.2f}%  "
          f"sharpe={spy_base['sharpe']:+.3f}  max_dd={spy_base['max_dd']*100:+.2f}%")

    # Best edge highlight
    print("\n" + "=" * 100)
    print("  TOP 5 CONFIGURATIONS BY EDGE OVER SPY")
    print("=" * 100)
    top5 = summary.sort_values("edge_vs_spy_pp", ascending=False).head(5)
    for _, r in top5.iterrows():
        verdict = "BEATS SPY" if r["edge_vs_spy_pp"] > 0 else "trails SPY"
        print(f"  {r['scheme']:<10} @ {int(r['bps'])}bps, borrow={int(r['borrow'])}bp: "
              f"ann={r['ann_ret']*100:+.2f}%  sharpe={r['sharpe']:+.3f}  "
              f"edge={r['edge_vs_spy_pp']:+.2f}pp  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
