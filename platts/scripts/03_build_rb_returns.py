"""Build a 3rd-prompt RB daily settlement + log-return series.

Reads per-contract CSVs from platts/data/{train,test}/futures/RB_*.csv and
constructs a continuous Nth-prompt series the same way refiner's
load_fixed_tenor_crack does -- but offline-only (no yfinance fill-in).

For each trading day, the 3rd-prompt contract is the third-nearest live
contract by expiry. Roll days (when the 3rd-prompt contract changes) are
flagged but not specially handled -- the log return on a roll day is the
ratio of the new contract's price today to the old contract's price
yesterday (mirrors refiner convention).

Output: platts/outputs/rb_returns.parquet with columns:
    date (index), settlement, log_return, contract_used, is_roll_day

Usage:
    python platts/scripts/03_build_rb_returns.py
    python platts/scripts/03_build_rb_returns.py --tenor 1   # front-month
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

DATA_DIRS = [
    PROJECT_ROOT / "platts" / "data" / "train" / "futures",
    PROJECT_ROOT / "platts" / "data" / "test"  / "futures",
]
OUT_PATH = PROJECT_ROOT / "platts" / "outputs" / "rb_returns.parquet"


def _load_contracts(product: str) -> list[dict]:
    """Load every RB_YYYY_MM.csv across train + test futures dirs."""
    contracts: list[dict] = []
    for d in DATA_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f"{product}_*.csv")):
            parts = f.stem.split("_")
            if len(parts) != 3:
                continue
            try:
                _, year, month = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            df = pd.read_csv(f, parse_dates=["date", "expiry_date"])
            if df.empty or "settlement" not in df.columns:
                continue
            close = (
                df.set_index("date")["settlement"]
                .dropna()
                .astype(float)
                .sort_index()
            )
            if close.empty:
                continue
            contracts.append({
                "symbol":   f.stem,
                "delivery": pd.Timestamp(year, month, 1),
                "expiry":   pd.Timestamp(df["expiry_date"].iloc[0]),
                "close":    close,
            })
    return contracts


def _build_nth_prompt(contracts: list[dict], tenor: int) -> pd.DataFrame:
    """For each trading day, pick the Nth-nearest live contract by expiry."""
    all_dates = sorted({d for c in contracts for d in c["close"].index})
    rows = []
    for d in all_dates:
        live = [c for c in contracts if c["expiry"] > d and d in c["close"].index]
        live.sort(key=lambda c: c["expiry"])
        if len(live) >= tenor:
            c = live[tenor - 1]
            rows.append({
                "date":           d,
                "settlement":     float(c["close"].loc[d]),
                "contract_used":  c["symbol"],
                "expiry":         c["expiry"],
                "days_to_expiry": (c["expiry"] - d).days,
            })
    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def build_rb_series(tenor: int) -> pd.DataFrame:
    print(f"[RB] Loading per-contract CSVs (tenor = {tenor}-prompt)...")
    contracts = _load_contracts("RB")
    if not contracts:
        raise RuntimeError(f"No RB contracts found in {DATA_DIRS}")
    print(f"     Found {len(contracts)} contracts spanning "
          f"{min(c['delivery'] for c in contracts).date()} to "
          f"{max(c['delivery'] for c in contracts).date()}")

    df = _build_nth_prompt(contracts, tenor)
    if df.empty:
        raise RuntimeError(
            f"Empty {tenor}-prompt series. Not enough overlapping contracts at "
            f"the requested tenor -- try a smaller --tenor value."
        )

    # Log return: ln(S_t / S_{t-1}). NaN on first day.
    df["log_return"] = np.log(df["settlement"] / df["settlement"].shift(1))

    # Roll-day flag: True when the contract used today differs from yesterday's
    df["is_roll_day"] = df["contract_used"] != df["contract_used"].shift(1)
    df.loc[df.index[0], "is_roll_day"] = False  # first day not a roll

    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tenor", type=int, default=3,
                   help="Nth-prompt to use (default 3 = third-nearest)")
    args = p.parse_args()

    df = build_rb_series(args.tenor)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, compression="snappy")

    # Summary
    print("\n" + "=" * 65)
    print(f"  RB {args.tenor}-prompt continuous series")
    print("=" * 65)
    print(f"  rows:           {len(df):,}")
    print(f"  date range:     {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  settle range:   ${df['settlement'].min():.4f} -> ${df['settlement'].max():.4f}")
    rets = df["log_return"].dropna()
    print(f"  return mean:    {rets.mean()*100:+.4f}%/day")
    print(f"  return std:     {rets.std()*100:.4f}%/day")
    print(f"  return ann vol: {rets.std()*np.sqrt(252)*100:.2f}%")
    print(f"  roll days:      {int(df['is_roll_day'].sum())}  "
          f"({df['is_roll_day'].mean()*100:.2f}% of days)")
    print(f"\n  Saved to: {OUT_PATH}")

    # Show first/last few rows
    print("\n  First 3 rows:")
    print(df.head(3).to_string())
    print("\n  Last 3 rows:")
    print(df.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
