"""Download real M3 (3rd-nearby) crack-spread futures from the EIA API.

The EIA publishes NYMEX-settled per-contract futures prices (contracts 1-4)
for WTI crude, NY Harbor RBOB gasoline and NY Harbor No.2 heating oil — daily,
free, back to the 1980s.  This is genuine forward-curve data (unlike yfinance,
which drops expired contracts), so it gives a consistent, real M3 crack across
the entire 2015-present period.

Output (a NEW folder, separate from data/cache):
    data/eia/futures_m3.csv   columns: date, CL, RB, HO, Crack_Spread

Usage:
    # free key: https://www.eia.gov/opendata/register.php
    export EIA_API_KEY=your_key_here
    VeefU9LYQ8pTmNC07xvHGdZphBCWNp0ruKoDUycm
    python scripts/00_download_eia_futures.py --start 2015-01-01 --end 2026-05-29

    # or pass the key inline:
    python scripts/00_download_eia_futures.py --api-key YOUR_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from refiner_strategy.config import EIA_DATA_DIR, EIA_SERIES_M3

EIA_ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
PAGE_LENGTH = 5000  # EIA v2 max rows per request


def fetch_series(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    """Fetch one EIA daily futures series as a date-indexed Series."""
    values: dict[pd.Timestamp, float] = {}
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series_id,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_LENGTH,
        }
        url = EIA_ENDPOINT + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)

        rows = payload.get("response", {}).get("data", [])
        for r in rows:
            v = r.get("value")
            if v is not None:
                values[pd.Timestamp(r["period"])] = float(v)

        if len(rows) < PAGE_LENGTH:
            break
        offset += PAGE_LENGTH

    return pd.Series(values, name=series_id).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EIA M3 crack futures")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None, help="default=today")
    parser.add_argument("--api-key", type=str, default=None, help="EIA API key (or set EIA_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        print(
            "ERROR: no EIA API key. Get a free key at\n"
            "  https://www.eia.gov/opendata/register.php\n"
            "then: export EIA_API_KEY=your_key   (or pass --api-key)"
        )
        sys.exit(1)

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    EIA_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading EIA M3 futures {args.start} → {end} into {EIA_DATA_DIR}\n")

    legs: dict[str, pd.Series] = {}
    for leg, series_id in EIA_SERIES_M3.items():
        print(f"  {leg} ({series_id})...", end=" ", flush=True)
        try:
            s = fetch_series(series_id, api_key, args.start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED ({exc})")
            sys.exit(1)
        if s.empty:
            print("no data returned — check the series ID at the EIA browser")
            sys.exit(1)
        legs[leg] = s
        print(f"{len(s)} rows, {s.index.min().date()} → {s.index.max().date()}")

    df = pd.DataFrame(legs).dropna()
    if df.empty:
        print("\nERROR: no overlapping dates across CL/RB/HO.")
        sys.exit(1)

    # 3:2:1 crack in $/bbl (RB, HO are $/gal → ×42).
    df["Crack_Spread"] = (2 * df["RB"] * 42 + df["HO"] * 42) / 3 - df["CL"]

    out_path = EIA_DATA_DIR / "futures_m3.csv"
    df.to_csv(out_path, index_label="date")
    print(f"\n✓ Saved {len(df)} rows to {out_path}")
    print(f"  Crack_Spread: mean={df['Crack_Spread'].mean():.2f}, "
          f"min={df['Crack_Spread'].min():.2f}, max={df['Crack_Spread'].max():.2f} $/bbl")


if __name__ == "__main__":
    main()
