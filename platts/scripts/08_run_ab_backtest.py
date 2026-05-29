"""A/B backtest: 6 sizing schemes on RB with PLATTS-derived DET.

Adapts refiner_strategy/evaluation/ab_runner.py for single-asset futures:
  - One product (RB) instead of a 7-stock basket
  - No beta hedge -- the asset IS the future, not a stock
  - Raw log returns instead of hedged returns
  - PLATTS-driver DET replaces crack-spread DET

Inputs:
    platts/outputs/master_dataset.parquet              (RB_LogReturn)
    platts/outputs/chronos_predictions/all_preds.parquet  (q10..q90, p_up)
    platts/outputs/platts_det_signal.parquet           (det_sig)

Output:
    platts/outputs/backtest/
        results_<bps>bps.csv     # scheme x metrics, one file per transaction cost
        pnl_curves_<bps>bps.csv  # date x scheme cumulative PnL
        summary.csv              # cross-bps summary

Usage:
    python platts/scripts/08_run_ab_backtest.py                # sweep [1,5,10] bps
    python platts/scripts/08_run_ab_backtest.py --bps 5
    python platts/scripts/08_run_ab_backtest.py --notional 100
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
MASTER_PATH   = PROJECT_ROOT / "platts" / "outputs" / "master_dataset.parquet"
PRED_PATH     = PROJECT_ROOT / "platts" / "outputs" / "chronos_predictions" / "all_preds.parquet"
DET_PATH      = PROJECT_ROOT / "platts" / "outputs" / "platts_det_signal.parquet"
SPY_PATH      = PROJECT_ROOT / "platts" / "data" / "train" / "SPY_daily.csv"
OUT_DIR       = PROJECT_ROOT / "platts" / "outputs" / "backtest"


def load_spy_returns() -> pd.Series:
    """Load SPY adjusted-close daily log returns from the train CSV."""
    spy = pd.read_csv(SPY_PATH, parse_dates=["date"])
    spy = spy.set_index("date").sort_index()
    spy.index = spy.index.normalize()
    return np.log(spy["Close"] / spy["Close"].shift(1)).dropna()

# ---------------------------------------------------------------------------
# Constants -- recalibrated from refiner for raw RB log returns (~1.5-2.5%/day)
# ---------------------------------------------------------------------------
TARGET_DAILY_VOL   = 0.015     # raw futures, not 0.01 hedged-equity
VOL_FLOOR          = 0.005
VOL_CAP            = 0.04      # 4% daily RV cap
DET_SIGNAL_MAG     = 0.02      # ~2% -- RB daily return scale
Q90_Q10_TO_SIGMA   = 2.5631    # 2 * 1.2816 (normal q10/q90 span)
OLD_DEADBAND_LOW   = 0.40
OLD_DEADBAND_HIGH  = 0.60
OLD_CONVICTION_SLOPE = 5.0
REALIZED_VOL_WIN   = 20
TRADING_DAYS       = 252

NEUTRAL_PRED = {
    "q10": 0.0, "q20": 0.0, "q30": 0.0, "q40": 0.0, "q50": 0.0,
    "q60": 0.0, "q70": 0.0, "q80": 0.0, "q90": 0.0, "p_up": 0.5,
}


# ===========================================================================
# Sizing schemes -- ported from refiner_strategy/sizing/schemes.py
# Signature: (pred_dict, det_sig, notional, realized_vol) -> dollar_size
# ===========================================================================

def _apply_vol_cap(base: float, rv: float | None) -> float:
    if rv is None or rv != rv or rv <= 0 or rv <= VOL_CAP:
        return base
    return base * (VOL_CAP / rv)


def size_old(pred, det_sig, notional, rv):
    p_up = pred.get("p_up", 0.5)
    if OLD_DEADBAND_LOW < p_up < OLD_DEADBAND_HIGH:
        return 0.0
    conviction = min(1.0, abs(p_up - 0.5) * OLD_CONVICTION_SLOPE)
    direction = 1.0 if p_up > 0.5 else -1.0
    return direction * conviction * notional


def size_new(pred, det_sig, notional, rv):
    q10, q50, q90 = pred.get("q10", 0.0), pred.get("q50", 0.0), pred.get("q90", 0.0)
    p_up = pred.get("p_up", 0.5)
    raw_fcst_vol = (q90 - q10) / Q90_Q10_TO_SIGMA
    if raw_fcst_vol < VOL_FLOOR:
        return 0.0
    if (q50 > 0) != (p_up > 0.5):
        return 0.0
    edge = q50 / raw_fcst_vol
    raw_size = edge / TARGET_DAILY_VOL
    clipped = max(-1.0, min(1.0, raw_size))
    return clipped * notional


def size_new_cap(pred, det_sig, notional, rv):
    return _apply_vol_cap(size_new(pred, det_sig, notional, rv), rv)


def size_det(pred, det_sig, notional, rv):
    if det_sig == 0:
        return 0.0
    return _apply_vol_cap(det_sig * notional, rv)


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
    edge = avg_q50 / raw_fcst_vol
    raw_size = edge / TARGET_DAILY_VOL
    clipped = max(-1.0, min(1.0, raw_size))
    return _apply_vol_cap(clipped * notional, rv)


SIZERS = {
    "OLD":       size_old,
    "NEW":       size_new,
    "NEW_CAP":   size_new_cap,
    "DET":       size_det,
    "ENS_VETO":  size_ens_veto,
    "ENS_AVG":   size_ens_avg,
}


# ===========================================================================
# Backtest engine
# ===========================================================================

def run_backtest(
    rb_returns: pd.Series,        # date -> log_return
    preds:      pd.DataFrame,      # date column + q10..q90 + p_up
    det_sig:    pd.Series,         # date -> ±1 or 0  (UNLAGGED; we shift here)
    schemes:    list[str],
    notional:   float,
    bps:        float,
    spy_returns: pd.Series,        # date -> log_return for SPY benchmark
) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    """Return (per-scheme cumulative PnL series, per-scheme metrics)."""
    # Lag det_sig by 1: today's signal used yesterday's close (T+1 convention)
    det_lagged = det_sig.shift(1).fillna(0)

    # Build O(1) prediction lookup
    preds_idx = preds.copy()
    preds_idx["Date"] = pd.to_datetime(preds_idx["Date"]).dt.normalize()
    preds_idx = preds_idx.set_index("Date")

    # Realized vol of RB log returns (trailing 20-day)
    rv = rb_returns.rolling(REALIZED_VOL_WIN, min_periods=5).std()

    # Find common dates: predictions ∩ RB ∩ DET
    dates = preds_idx.index.intersection(rb_returns.index).intersection(det_lagged.index)
    dates = sorted(dates)
    if not dates:
        raise RuntimeError("No overlapping dates between preds, RB returns, and DET.")

    # Per-scheme state
    state = {s: {"position": 0.0, "pnl": [], "trades_count": 0,
                 "hits": 0, "active_days": 0, "txn_total": 0.0} for s in schemes}

    txn_rate = bps / 10_000  # round-trip bps; applied per |position change|

    for T in dates:
        # Build pred dict for today
        if T in preds_idx.index:
            row = preds_idx.loc[T]
            pred = {k: float(row[k]) for k in NEUTRAL_PRED if k in row.index}
        else:
            pred = dict(NEUTRAL_PRED)
        # Sanitize NaNs
        pred = {k: (v if v == v else NEUTRAL_PRED[k]) for k, v in pred.items()}

        det_today = float(det_lagged.loc[T])
        rv_today  = rv.loc[T] if T in rv.index else None
        if rv_today is not None and rv_today != rv_today:
            rv_today = None
        ret_today = float(rb_returns.loc[T])

        for scheme in schemes:
            sizer = SIZERS[scheme]
            target = sizer(pred, det_today, notional, rv_today)
            effective = state[scheme]["position"]      # what we held coming in

            friction = abs(target - effective) * txn_rate
            asset_pnl = effective * ret_today - friction

            state[scheme]["pnl"].append((T, asset_pnl))
            state[scheme]["txn_total"] += friction
            if target != effective:
                state[scheme]["trades_count"] += 1
            if effective != 0:
                state[scheme]["active_days"] += 1
                if math.copysign(1, effective) == math.copysign(1, ret_today) and ret_today != 0:
                    state[scheme]["hits"] += 1
            state[scheme]["position"] = target

    # Build PnL series + metrics
    pnl_series = {}
    metrics = {}
    for s in schemes:
        pdf = pd.DataFrame(state[s]["pnl"], columns=["date", "pnl"]).set_index("date")
        pnl = pdf["pnl"]
        cum = pnl.cumsum()
        pnl_series[s] = pnl
        m = _compute_metrics(pnl, notional)
        m["txn_cost_total"] = state[s]["txn_total"]
        m["num_trades"]     = state[s]["trades_count"]
        m["active_days"]    = state[s]["active_days"]
        m["hit_rate"] = (state[s]["hits"] / state[s]["active_days"]
                        if state[s]["active_days"] > 0 else float("nan"))
        m["cum_pnl"]   = float(cum.iloc[-1])
        m["bps"]       = bps
        metrics[s] = m

    # Benchmarks (compute fresh)
    rb_bh_pnl = notional * rb_returns.loc[dates]   # long-only RB, no friction
    spy_aligned = spy_returns.reindex(dates).fillna(0)
    spy_bh_pnl = notional * spy_aligned             # long-only SPY, no friction
    pnl_series["RB_BH"]  = rb_bh_pnl
    pnl_series["SPY_BH"] = spy_bh_pnl
    metrics["RB_BH"]  = _compute_metrics(rb_bh_pnl,  notional)
    metrics["SPY_BH"] = _compute_metrics(spy_bh_pnl, notional)

    return pnl_series, metrics


def _compute_metrics(pnl: pd.Series, notional: float) -> dict:
    """Sharpe, ann return, max DD, vol."""
    pnl = pnl.dropna()
    if pnl.empty or pnl.std() == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "max_dd": 0.0, "total_pnl": 0.0}
    mean_d = pnl.mean()
    std_d  = pnl.std()
    sharpe = (mean_d / std_d) * math.sqrt(TRADING_DAYS) if std_d > 0 else 0.0
    ann_ret = mean_d * TRADING_DAYS / notional        # % per year
    ann_vol = std_d * math.sqrt(TRADING_DAYS) / notional
    cum = pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min() / notional
    return {
        "sharpe":     float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol":    float(ann_vol),
        "max_dd":     float(max_dd),
        "total_pnl":  float(pnl.sum()),
    }


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bps", type=float, nargs="+", default=[1.0, 5.0, 10.0],
                   help="Round-trip transaction cost(s) in bps; can list multiple")
    p.add_argument("--notional", type=float, default=100.0,
                   help="Dollar notional per scheme (default 100)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print(f"[LOAD] {MASTER_PATH.name} ...")
    master = pd.read_parquet(MASTER_PATH)
    master.index = pd.to_datetime(master.index).normalize()
    rb_ret = master["RB_LogReturn"]

    print(f"[LOAD] {PRED_PATH.name} ...")
    preds = pd.read_parquet(PRED_PATH)
    print(f"       {len(preds)} predictions, "
          f"{preds['Date'].min()} -> {preds['Date'].max()}")

    print(f"[LOAD] {DET_PATH.name} ...")
    det = pd.read_parquet(DET_PATH)
    det.index = pd.to_datetime(det.index).normalize()
    det_sig = det["det_sig"]

    print(f"[LOAD] {SPY_PATH.name} ...")
    spy_ret = load_spy_returns()
    print(f"       SPY: {len(spy_ret):,} rows, "
          f"{spy_ret.index.min().date()} -> {spy_ret.index.max().date()}")

    schemes = list(SIZERS.keys())
    print(f"\n[CONFIG] schemes = {schemes}")
    print(f"         notional = ${args.notional}")
    print(f"         bps sweep = {args.bps}\n")

    # Sweep over transaction costs
    summary_rows = []
    for bps in args.bps:
        print(f"{'='*78}")
        print(f"  BACKTEST @ {bps:.1f} bps round-trip")
        print(f"{'='*78}")
        pnl_series, metrics = run_backtest(
            rb_ret, preds, det_sig, schemes, args.notional, bps,
            spy_returns=spy_ret,
        )

        # Save per-bps results
        results = pd.DataFrame(metrics).T
        results.index.name = "scheme"
        results_path = OUT_DIR / f"results_{int(bps)}bps.csv"
        results.to_csv(results_path)

        pnl_df = pd.DataFrame({s: pnl_series[s] for s in pnl_series}).sort_index()
        cum_df = pnl_df.cumsum()
        cum_df.to_csv(OUT_DIR / f"pnl_curves_{int(bps)}bps.csv")

        # Print table
        print(f"\n  {'scheme':<10}  {'sharpe':>7}  {'ann_ret':>8}  {'ann_vol':>7}  "
              f"{'max_dd':>8}  {'hit_rate':>9}  {'trades':>7}  {'txn_$':>7}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}")
        for s in schemes + ["RB_BH", "SPY_BH"]:
            m = metrics[s]
            hr = m.get("hit_rate", float("nan"))
            hr_str = f"{hr*100:7.2f}%" if hr == hr else "    n/a"
            nt = m.get("num_trades", 0)
            txn = m.get("txn_cost_total", 0.0)
            print(f"  {s:<10}  {m['sharpe']:>+7.3f}  {m['ann_return']*100:>+7.2f}%  "
                  f"{m['ann_vol']*100:>6.2f}%  {m['max_dd']*100:>+7.2f}%  "
                  f"{hr_str:>9}  {nt:>7d}  ${txn:>6.2f}")

        for s, m in metrics.items():
            # Spread m first so {bps, scheme} always win the override
            row = {**m, "bps": bps, "scheme": s}
            summary_rows.append(row)
        print(f"\n  Wrote {results_path.name} + pnl_curves_{int(bps)}bps.csv")

    # Cross-bps summary
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Combined summary -> {summary_path}")

    # Verdict highlights
    print("\n" + "=" * 78)
    print("  HEADLINE: best scheme per bps level (Sharpe), vs RB and SPY buy-and-hold")
    print("=" * 78)
    for bps in args.bps:
        rows = summary[summary["bps"] == bps]
        non_bench = rows[~rows["scheme"].isin(["RB_BH", "SPY_BH"])]
        best = non_bench.loc[non_bench["sharpe"].idxmax()]
        rb_bh  = rows[rows["scheme"] == "RB_BH"].iloc[0]
        spy_bh = rows[rows["scheme"] == "SPY_BH"].iloc[0]
        diff_rb  = best["ann_return"] - rb_bh["ann_return"]
        diff_spy = best["ann_return"] - spy_bh["ann_return"]
        v_rb  = "BEATS" if diff_rb  > 0 else "TRAILS"
        v_spy = "BEATS" if diff_spy > 0 else "TRAILS"
        print(f"  @ {bps:.0f}bps:  best={best['scheme']:<9}  "
              f"sharpe={best['sharpe']:+.3f}  ann={best['ann_return']*100:+.2f}%")
        print(f"            vs RB_BH  ({rb_bh['ann_return']*100:+.2f}%):  "
              f"{diff_rb*100:+.2f}pp  {v_rb}")
        print(f"            vs SPY_BH ({spy_bh['ann_return']*100:+.2f}%):  "
              f"{diff_spy*100:+.2f}pp  {v_spy}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
