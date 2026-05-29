"""Download and cache yfinance data locally for reproducible strategy runs.

This script fetches futures contracts (CL/RB/HO) and equity prices (refiners + SPY)
from yfinance and saves them as CSVs in the local cache directory. Subsequent strategy
runs use these cached files instead of hitting yfinance repeatedly, ensuring
reproducibility and faster iteration.

Usage:
    python scripts/00_download_and_cache_data.py --start 2022-01-01 --end 2026-05-06
    python scripts/00_download_and_cache_data.py  # defaults: 2015-01-01 to today
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import yfinance as yf

from refiner_strategy.config import TICKERS


CACHE_DIR = PROJECT_DIR / "data" / "cache"
CACHE_FUTURES_DIR = CACHE_DIR / "futures"


def _month_code(month: int) -> str:
    """CME month code from month number (1=F, 2=G, etc.)."""
    codes = {
        1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
        7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
    }
    return codes[month]


# Minimum rows for a contract to be considered real (yfinance returns a stale
# 1-row stub for delisted/expired contracts — those are skipped).
MIN_CONTRACT_ROWS = 20

# Expiry-day approximation (day of the month *preceding* delivery).
_EXPIRY_DAY = {"CL": 20, "RB": 25, "HO": 25}


def _contract_expiry(product: str, year: int, month: int) -> pd.Timestamp:
    """Approximate last-trade date: a day in the month preceding delivery."""
    day = _EXPIRY_DAY[product]
    if month == 1:
        return pd.Timestamp(year - 1, 12, day)
    return pd.Timestamp(year, month - 1, day)


def _save_contract(product: str, year: int, month: int, s: pd.Series) -> bool:
    """Write one contract CSV in bundle-compatible schema. Returns True if saved."""
    s = s.dropna()
    if len(s) < MIN_CONTRACT_ROWS:
        return False
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    expiry = _contract_expiry(product, year, month)
    df = pd.DataFrame(
        {
            "date": s.index,
            "settlement": s.values.astype(float),
            "expiry_date": expiry,
            "days_to_expiry": (expiry - s.index).days,
        }
    )
    filename = f"{product}_{year}_{month:02d}.csv"
    df.to_csv(CACHE_FUTURES_DIR / filename, index=False)
    return True


def download_futures(start: str, end: str) -> None:
    """Download CL/RB/HO dated contracts + continuous front-month, save to cache.

    yfinance only carries *currently-listed* contracts — expired ones return an
    empty/stub series.  We save only contracts with real data (skipping stubs)
    and ALSO cache the continuous front-month (CL=F/RB=F/HO=F), which has a price
    every trading day and serves as the gap-free recent source when per-contract
    M6 cannot be reconstructed.
    """
    CACHE_FUTURES_DIR.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    print(f"Downloading dated futures contracts (CL/RB/HO) for {start} to {end}...")

    contracts_info = []
    for product in ["CL", "RB", "HO"]:
        for year in range(start_ts.year - 1, end_ts.year + 3):
            for month in range(1, 13):
                symbol = f"{product}{_month_code(month)}{str(year)[-2:]}.NYM"
                contracts_info.append(
                    {"symbol": symbol, "product": product, "year": year, "month": month}
                )

    saved, skipped_empty = 0, 0
    batch_size = 50
    for i in range(0, len(contracts_info), batch_size):
        batch = contracts_info[i : i + batch_size]
        symbols = [c["symbol"] for c in batch]
        print(f"  Batch {i // batch_size + 1}/{(len(contracts_info) - 1) // batch_size + 1}...", end=" ", flush=True)

        raw = yf.download(
            symbols, start=start, end=end, progress=False,
            auto_adjust=False, group_by="ticker", threads=True,
        )
        batch_saved = 0
        if not raw.empty:
            for info in batch:
                try:
                    if len(symbols) == 1:
                        s = raw["Close"] if "Close" in raw.columns else None
                    else:
                        s = raw[info["symbol"]]["Close"] if info["symbol"] in raw.columns else None
                    if s is None:
                        continue
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    if _save_contract(info["product"], info["year"], info["month"], s):
                        batch_saved += 1
                    else:
                        skipped_empty += 1
                except (KeyError, AttributeError):
                    skipped_empty += 1
        saved += batch_saved
        print(f"(saved {batch_saved})")

    print(f"  → {saved} contracts with real data, {skipped_empty} empty/delisted skipped")

    # Continuous front-month (gap-free recent fallback source).
    print("Downloading continuous front-month (CL=F, RB=F, HO=F)...")
    cont = yf.download(["CL=F", "RB=F", "HO=F"], start=start, end=end,
                       progress=False, auto_adjust=False)
    if not cont.empty:
        close = cont["Close"].copy()
        close.columns = ["CL", "HO", "RB"]  # yfinance alphabetical order
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        close.dropna().to_csv(CACHE_DIR / "continuous_front_month.csv")
        print(f"  Saved continuous front-month: {len(close.dropna())} rows")

    print(f"Futures cached in {CACHE_FUTURES_DIR}\n")


def download_equities(start: str, end: str) -> None:
    """Download equity prices and save to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    symbols = TICKERS + ["SPY"]
    print(f"Downloading equities {symbols} for {start} to {end}...")

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if raw.empty:
        print("No data downloaded")
        return

    if len(symbols) == 1:
        close = raw["Close"].to_frame(symbols[0])
    else:
        close = raw["Close"].copy()

    for symbol in symbols:
        if symbol in close.columns:
            df = close[[symbol]].reset_index()
            df.columns = ["date", "Close"]
            filepath = CACHE_DIR / f"{symbol}_daily.csv"
            df.to_csv(filepath, index=False)
            print(f"  Saved {symbol}: {len(df)} rows")

    print(f"Equities cached in {CACHE_DIR}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download yfinance data and cache locally for reproducible strategy runs"
    )
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD); default=today")
    parser.add_argument("--futures-only", action="store_true", help="Download only futures")
    parser.add_argument("--equities-only", action="store_true", help="Download only equities")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    print(f"Cache directory: {CACHE_DIR}\n")

    if not args.equities_only:
        download_futures(args.start, args.end)
    if not args.futures_only:
        download_equities(args.start, args.end)

    print("✓ Data cached. Subsequent runs will use local files.")
    print(f"  To update cache, run: python scripts/00_download_and_cache_data.py --start <date>")


if __name__ == "__main__":
    main()
