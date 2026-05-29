"""Assemble the Chronos-ready master dataset.

Joins:
    platts/outputs/rb_returns.parquet                         (Y target)
    platts/outputs/pricedata/zscore/pricedata_ml_ready.parquet (X candidates)
    platts/outputs/moirai_rb_ranking.csv                       (top-K selection)

For each of the top-K MOIRAI-ranked PLATTS symbols we pull its Z_SCORE
column (already 256-day rolling per the local pipeline) and align on RB's
trading-day calendar. Weekends and PLATTS-only holidays are dropped.

Output: platts/outputs/master_dataset.parquet with columns:
    date (index),
    RB_LogReturn      <- Y target
    RB_Settlement     <- raw price for reference
    {SYMBOL}_Z        <- top-K X covariates (one per MOIRAI winner)

Usage:
    python platts/scripts/05_build_master_dataset.py
    python platts/scripts/05_build_master_dataset.py --top-k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

ZSCORE_PATH   = PROJECT_ROOT / "platts" / "outputs" / "pricedata" / "zscore" / "pricedata_ml_ready.parquet"
RB_PATH       = PROJECT_ROOT / "platts" / "outputs" / "rb_returns.parquet"
RANK_PATH     = PROJECT_ROOT / "platts" / "outputs" / "moirai_rb_ranking.csv"
OUT_PATH      = PROJECT_ROOT / "platts" / "outputs" / "master_dataset.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=5, help="MOIRAI rank cutoff (default 5)")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Top-K MOIRAI symbols
    # ------------------------------------------------------------------
    print(f"[1/4] Loading top-{args.top_k} MOIRAI ranking ...")
    rank = pd.read_csv(RANK_PATH)
    if rank.empty:
        print(f"[FAIL] {RANK_PATH} is empty. Re-run 04_moirai_rb_discovery.py first.")
        return 1
    top = rank.head(args.top_k).copy()
    top_symbols = top["SYMBOL"].tolist()
    print(f"       selected: {top_symbols}")

    # ------------------------------------------------------------------
    # 2. PLATTS Z-scores for those symbols only (avoid full 88M-row load)
    # ------------------------------------------------------------------
    print(f"[2/4] Loading PLATTS Z-scores for {len(top_symbols)} symbols ...")
    # pyarrow filter is faster than read-then-filter for sparse selection
    import pyarrow.parquet as pq
    tbl = pq.read_table(
        ZSCORE_PATH,
        columns=["SYMBOL", "ASSESSDATE", "Z_SCORE"],
        filters=[("SYMBOL", "in", top_symbols)],
    )
    platts = tbl.to_pandas()
    platts["ASSESSDATE"] = pd.to_datetime(platts["ASSESSDATE"]).dt.normalize()
    print(f"       {len(platts):,} rows across {platts['SYMBOL'].nunique()} symbols")

    wide = (platts.pivot(index="ASSESSDATE", columns="SYMBOL", values="Z_SCORE")
                  .sort_index())
    wide.columns = [f"{c}_Z" for c in wide.columns]

    # ------------------------------------------------------------------
    # 3. RB returns (Y target)
    # ------------------------------------------------------------------
    print("[3/4] Loading RB returns ...")
    rb = pd.read_parquet(RB_PATH)
    rb.index = pd.to_datetime(rb.index).normalize()
    rb = rb[["settlement", "log_return"]].rename(columns={
        "settlement": "RB_Settlement",
        "log_return": "RB_LogReturn",
    })
    print(f"       {len(rb):,} RB rows, {rb.index.min().date()} -> {rb.index.max().date()}")

    # ------------------------------------------------------------------
    # 4. Align on RB's trading-day calendar (inner join on dates)
    # ------------------------------------------------------------------
    print("[4/4] Aligning on RB trading-day index ...")
    master = rb.join(wide, how="left").sort_index()
    master = master.dropna(subset=["RB_LogReturn"])

    # Forward-fill PLATTS gaps that are weekends/PLATTS-holidays falling on
    # trading days (rare but happens). Don't bfill — that's look-ahead.
    z_cols = [c for c in master.columns if c.endswith("_Z")]
    master[z_cols] = master[z_cols].ffill()

    # Drop any rows where covariates are still NaN (typically only the
    # very first day before any PLATTS data has arrived).
    before = len(master)
    master = master.dropna(subset=z_cols)
    dropped = before - len(master)
    if dropped:
        print(f"       Dropped {dropped} rows with missing PLATTS Z (pre-coverage warmup)")

    # ------------------------------------------------------------------
    # Save + summary
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUT_PATH, compression="snappy")
    print(f"\n[OK] Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    print("\n" + "=" * 78)
    print("  MASTER DATASET SUMMARY")
    print("=" * 78)
    print(f"  rows:        {len(master):,}")
    print(f"  date range:  {master.index.min().date()} -> {master.index.max().date()}")
    print(f"  columns:     {list(master.columns)}")

    print("\n  Covariate metadata (from MOIRAI ranking):")
    print(f"  {'RANK':>4}  {'SYMBOL':<10}  {'SCORE':>7}  {'PRODUCT':<30}  GRADE")
    print(f"  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*30}  {'-'*20}")
    for _, row in top.iterrows():
        prod  = (str(row.get("PRODUCT") or "")[:30])
        grade = (str(row.get("GRADE")   or "")[:20])
        print(f"  {int(row['RANK']):>4}  {row['SYMBOL']:<10}  "
              f"{row['ATTENTION_FROM_RB']:>7.4f}  {prod:<30}  {grade}")

    print("\n  Pairwise correlations with RB_LogReturn:")
    corrs = master[z_cols].corrwith(master["RB_LogReturn"]).sort_values(key=abs, ascending=False)
    for c, v in corrs.items():
        print(f"    {c:<14}  {v:+.4f}")

    print("\n  First 3 rows:")
    print(master.head(3).to_string())
    print("\n  Last 3 rows:")
    print(master.tail(3).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
