"""Build a PLATTS-derived deterministic signal for RB.

Mirrors refiner_strategy/signals/det_signal.py structure but replaces the
crack-spread input with the top-1 MOIRAI-ranked PLATTS Z-score:

    raw_sig[t] = +1  if  Z[t]  >  SMA10(Z)[t]
                 -1  if  Z[t]  <  SMA10(Z)[t]
                  0  otherwise

    det_sig[t] = raw_sig[t]   if  raw_sig[t] == raw_sig[t-1]  (2-day persistence)
                 0           otherwise

The persistence filter prevents whipsaws. det_sig is UNLAGGED -- the
A/B harness shifts by 1 day at consumption.

Inputs:
    platts/outputs/moirai_rb_ranking.csv       (top-1 driver symbol)
    platts/outputs/pricedata/zscore/pricedata_ml_ready.parquet  (Z-scores)
    platts/outputs/rb_returns.parquet          (RB trading-day calendar)

Output:
    platts/outputs/platts_det_signal.parquet
        columns: date (index), driver_symbol, Z, SMA10, raw_sig, det_sig

Usage:
    python platts/scripts/07_build_det_signal.py
    python platts/scripts/07_build_det_signal.py --rank 1 --sma 10 --confirm 2
    python platts/scripts/07_build_det_signal.py --rank 1,2,3   # composite of top-3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

ZSCORE_PATH = PROJECT_ROOT / "platts" / "outputs" / "pricedata" / "zscore" / "pricedata_ml_ready.parquet"
RB_PATH     = PROJECT_ROOT / "platts" / "outputs" / "rb_returns.parquet"
RANK_PATH   = PROJECT_ROOT / "platts" / "outputs" / "moirai_rb_ranking.csv"
OUT_PATH    = PROJECT_ROOT / "platts" / "outputs" / "platts_det_signal.parquet"


def _confirm(raw: pd.Series, n: int) -> pd.Series:
    """N-day persistence filter: signal survives only if equal to its n-1 prior values."""
    keep = raw.copy()
    for i in range(1, n):
        keep = keep.where(raw == raw.shift(i), 0)
    return keep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", default="1",
                   help="Which MOIRAI rank(s) to use. '1' = top driver only, "
                        "'1,2,3' = attention-weighted composite of top-3.")
    p.add_argument("--sma",     type=int, default=10, help="SMA window (default 10)")
    p.add_argument("--confirm", type=int, default=2,  help="Persistence days (default 2)")
    args = p.parse_args()

    ranks = [int(x.strip()) for x in args.rank.split(",")]
    print(f"[CONFIG] ranks={ranks}  sma={args.sma}  confirm={args.confirm}\n")

    # ------------------------------------------------------------------
    # 1. Resolve driver symbols from MOIRAI ranking
    # ------------------------------------------------------------------
    print(f"[1/4] Loading MOIRAI ranking ...")
    rank_df = pd.read_csv(RANK_PATH)
    if rank_df.empty:
        print(f"[FAIL] {RANK_PATH} is empty. Re-run 04_moirai_rb_discovery.py first.")
        return 1

    chosen = rank_df[rank_df["RANK"].isin(ranks)].copy()
    if chosen.empty:
        print(f"[FAIL] No symbols match rank(s) {ranks}")
        return 1
    drivers   = chosen["SYMBOL"].tolist()
    weights   = chosen["ATTENTION_FROM_RB"].values
    weights   = weights / weights.sum()  # normalize for composite
    print(f"       drivers: {drivers}")
    for s, w in zip(drivers, weights):
        print(f"         {s}  weight={w:.3f}")

    # ------------------------------------------------------------------
    # 2. Load Z-scores for those symbols only
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading PLATTS Z-scores for {len(drivers)} symbol(s) ...")
    import pyarrow.parquet as pq
    tbl = pq.read_table(
        ZSCORE_PATH,
        columns=["SYMBOL", "ASSESSDATE", "Z_SCORE"],
        filters=[("SYMBOL", "in", drivers)],
    )
    platts = tbl.to_pandas()
    platts["ASSESSDATE"] = pd.to_datetime(platts["ASSESSDATE"]).dt.normalize()
    wide = (platts.pivot(index="ASSESSDATE", columns="SYMBOL", values="Z_SCORE")
                  .sort_index())
    print(f"       {len(wide):,} dates x {wide.shape[1]} symbols")

    # ------------------------------------------------------------------
    # 3. Composite Z = attention-weighted mean across selected drivers
    # ------------------------------------------------------------------
    if len(drivers) == 1:
        composite_z = wide[drivers[0]]
    else:
        weight_map = dict(zip(drivers, weights))
        composite_z = sum(wide[s].fillna(0) * weight_map[s] for s in drivers)
        composite_z.name = "composite_Z"

    # ------------------------------------------------------------------
    # 4. SMA, raw signal, persistence-confirmed signal
    # ------------------------------------------------------------------
    print(f"\n[3/4] Computing SMA{args.sma} + signal ...")
    sma = composite_z.rolling(args.sma, min_periods=max(args.sma // 2, 1)).mean()
    raw_sig = pd.Series(0.0, index=composite_z.index)
    raw_sig[composite_z > sma] =  1.0
    raw_sig[composite_z < sma] = -1.0
    det_sig = _confirm(raw_sig, args.confirm)

    df = pd.DataFrame({
        "Z":         composite_z,
        f"SMA{args.sma}": sma,
        "raw_sig":   raw_sig,
        "det_sig":   det_sig,
    })

    # ------------------------------------------------------------------
    # Align to RB trading-day calendar for downstream A/B harness
    # ------------------------------------------------------------------
    print(f"\n[4/4] Aligning to RB trading-day calendar ...")
    rb = pd.read_parquet(RB_PATH)
    rb.index = pd.to_datetime(rb.index).normalize()
    df = df.reindex(rb.index).ffill()
    # det_sig only valid after warmup + after first persistence-confirm window
    df["det_sig"] = df["det_sig"].fillna(0)
    df["raw_sig"] = df["raw_sig"].fillna(0)
    df["driver"] = ",".join(drivers)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, compression="snappy")
    print(f"\n[OK] Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e3:.1f} KB)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    n_long   = int((df["det_sig"] ==  1).sum())
    n_short  = int((df["det_sig"] == -1).sum())
    n_flat   = int((df["det_sig"] ==  0).sum())
    n_total  = len(df)

    print("\n" + "=" * 78)
    print(f"  DET SIGNAL SUMMARY  (driver: {','.join(drivers)})")
    print("=" * 78)
    print(f"  dates:       {df.index.min().date()} -> {df.index.max().date()}  ({n_total} rows)")
    print(f"  det_sig +1:  {n_long:>5} days  ({n_long /n_total*100:5.1f}%)")
    print(f"  det_sig -1:  {n_short:>5} days  ({n_short/n_total*100:5.1f}%)")
    print(f"  det_sig  0:  {n_flat:>5} days  ({n_flat /n_total*100:5.1f}%)")
    print(f"  time in mkt: {(n_long+n_short)/n_total*100:.1f}%")

    # Naive correlation: det_sig(t-1) vs RB return(t)  -- hint at predictive power
    lagged_sig = df["det_sig"].shift(1)
    rb_ret = rb["log_return"]
    common = lagged_sig.dropna().index.intersection(rb_ret.dropna().index)
    if len(common) > 30:
        # Hit rate when signal is non-zero
        active = lagged_sig.loc[common] != 0
        if active.sum() > 0:
            agree = (np.sign(rb_ret.loc[common]) == np.sign(lagged_sig.loc[common])) & active
            hit_rate = agree.sum() / active.sum()
            print(f"\n  Naive directional hit rate (sig_{{t-1}} vs RB_ret_t): "
                  f"{hit_rate*100:.2f}%  on {int(active.sum())} active days")
            print(f"  (>50% = useful; <50% = consider flipping the signal)")

    print("\n  Sample (first 5 active days):")
    active_rows = df[df["det_sig"] != 0].head(5)
    print(active_rows.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
