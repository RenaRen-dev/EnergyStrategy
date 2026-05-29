"""Download real M3 (3rd-nearby) crack-spread futures from Databento.

Databento serves official CME Globex data (the venue NYMEX WTI/RBOB/HO settle
on).  Its continuous-contract symbology lets us request a constant-maturity
3rd-nearby series directly — ``CL.c.2`` / ``RB.c.2`` / ``HO.c.2`` (rank 2 =
3rd-nearby, calendar roll).  Unlike EIA (discontinued April 2024) or yfinance
(drops expired contracts), this gives a real, consistent M3 across the full
period from a single source.

Output (a NEW folder, separate from data/cache and data/eia):
    data/databento/futures_m3.csv   columns: date, CL, RB, HO, Crack_Spread

Usage:
    pip install databento
    # free key + credits: https://databento.com
    export DATABENTO_API_KEY=db-XXXXXXXX
    python scripts/00_download_databento_futures.py --start 2015-01-01 --end 2026-05-29
"""

# # $VENV scripts/00_download_databento_futures.py --start 2024-03-01 --end 2026-05-27

from __future__ import annotations

import argparse
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import time
import warnings

import pandas as pd

from refiner_strategy.config import (
    DATABENTO_DATASET,
    DATABENTO_DATA_DIR,
    DATABENTO_SYMBOLS_M3,
)

# Errors that should NOT be retried locally (the caller retreats the end date).
_FATAL_MARKERS = ("unavailable_range", "422")

# Months per request: small enough that a dropped stream only re-fetches a few
# months.  3-year chunks stall reliably on long streams; 6-month chunks complete
# in a few seconds even on a slow connection.
_CHUNK_MONTHS = 6

# Hard per-request timeout (seconds): a daily-bar 6-month chunk should return in
# well under 60s, so anything past this is a stalled connection — abort and retry.
_REQUEST_TIMEOUT = 120


class _RequestTimeout(Exception):
    """Raised when a single Databento request exceeds _REQUEST_TIMEOUT."""


def _alarm_handler(signum, frame):  # noqa: ANN001, D401
    raise _RequestTimeout()


def _fetch_chunk(client, symbol: str, start: str, end: str, max_retries: int = 5) -> pd.Series:
    """Fetch one [start, end) chunk of daily closes, retrying transient errors.

    Streaming drops ("Response ended prematurely") and similar transient network
    errors are retried with backoff.  A 422/unavailable_range is re-raised so the
    caller can retreat the overall end date.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    for attempt in range(max_retries):
        try:
            if use_alarm:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(_REQUEST_TIMEOUT)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # silence BentoWarning (degraded days)
                    data = client.timeseries.get_range(
                        dataset=DATABENTO_DATASET,
                        symbols=[symbol],
                        stype_in="continuous",
                        schema="ohlcv-1d",
                        start=start,
                        end=end,
                    )
                    df = data.to_df()
            finally:
                if use_alarm:
                    signal.alarm(0)  # always clear the alarm
            if df.empty:
                return pd.Series(dtype=float, name=symbol)
            close = df["close"].copy()
            idx = pd.to_datetime(close.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            close.index = idx.normalize()
            return close[~close.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # noqa: BLE001
            if any(m in str(exc) for m in _FATAL_MARKERS):
                raise
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                reason = "timeout" if isinstance(exc, _RequestTimeout) else type(exc).__name__
                print(f"[retry {attempt + 1}/{max_retries} after {reason}, waiting {wait}s]",
                      end=" ", flush=True)
                time.sleep(wait)
                continue
            raise


def fetch_continuous_close(client, leg: str, symbol: str, start: str, end: str) -> pd.Series:
    """Fetch daily close for one continuous symbol (e.g. CL.c.2), in chunks.

    Chunking keeps each stream small so a dropped connection only re-fetches a
    few years.  Progress is printed per chunk so a slow/stalled request is visible.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    pieces: list[pd.Series] = []
    cur = start_ts
    while cur < end_ts:
        # Advance by _CHUNK_MONTHS calendar months.
        next_month = cur.month + _CHUNK_MONTHS
        next_year  = cur.year + (next_month - 1) // 12
        next_month = (next_month - 1) % 12 + 1
        chunk_end  = min(pd.Timestamp(year=next_year, month=next_month, day=1), end_ts)
        print(f"    {leg} {cur.date()} → {chunk_end.date()} ...", end=" ", flush=True)
        t0 = time.time()
        s = _fetch_chunk(
            client, symbol, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        )
        print(f"{len(s)} rows ({time.time() - t0:.0f}s)", flush=True)
        if not s.empty:
            pieces.append(s)
        cur = chunk_end

    if not pieces:
        return pd.Series(dtype=float, name=symbol)
    out = pd.concat(pieces)
    return out[~out.index.duplicated(keep="last")].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Databento M3 crack futures")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None, help="default=yesterday")
    parser.add_argument("--api-key", type=str, default=None, help="Databento key (or set DATABENTO_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print(
            "ERROR: no Databento API key. Get a free key (with credits) at\n"
            "  https://databento.com\n"
            "then: export DATABENTO_API_KEY=db-XXXX   (or pass --api-key)"
        )
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("ERROR: databento not installed. Run:  pip install databento")
        sys.exit(1)

    # Default to yesterday: Databento historical data lags the live session, so
    # requesting "today" trips a dataset_unavailable_range (422) error.
    end = args.end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    DATABENTO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = db.Historical(api_key)

    print(f"Downloading Databento M3 futures {args.start} → {end} into {DATABENTO_DATA_DIR}")
    print(f"  dataset={DATABENTO_DATASET}, symbols={list(DATABENTO_SYMBOLS_M3.values())}")

    # Fast connectivity + coverage probe (confirms key/network work before the
    # bulk download, and shows the dataset's real available date range).
    try:
        rng = client.metadata.get_dataset_range(dataset=DATABENTO_DATASET)
        print(f"  dataset available range: {rng}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read dataset range ({exc})\n")

    # Fetch all legs, auto-retreating the end date if it's too recent (422).
    legs: dict[str, pd.Series] = {}
    for attempt in range(8):
        try:
            legs = {
                leg: fetch_continuous_close(client, leg, symbol, args.start, end)
                for leg, symbol in DATABENTO_SYMBOLS_M3.items()
            }
            break
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "unavailable_range" in msg or "422" in msg:
                new_end = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"  end {end} too recent for historical data; retrying with {new_end}")
                end = new_end
                continue
            print(f"FAILED ({exc})")
            sys.exit(1)
    else:
        print("ERROR: could not find an available end date after several retries.")
        sys.exit(1)

    for leg, s in legs.items():
        if s.empty:
            print(f"  {leg}: no data — check the symbol / dataset coverage")
            sys.exit(1)
        print(f"  {leg} ({DATABENTO_SYMBOLS_M3[leg]}): {len(s)} rows, "
              f"{s.index.min().date()} → {s.index.max().date()}")

    df = pd.DataFrame(legs).dropna()
    if df.empty:
        print("\nERROR: no overlapping dates across CL/RB/HO.")
        sys.exit(1)

    # 3:2:1 crack in $/bbl (RB, HO are $/gal → ×42).
    df["Crack_Spread"] = (2 * df["RB"] * 42 + df["HO"] * 42) / 3 - df["CL"]

    out_path = DATABENTO_DATA_DIR / "futures_m3.csv"
    df.to_csv(out_path, index_label="date")
    print(f"\n✓ Saved {len(df)} rows to {out_path}")
    print(f"  range {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Crack_Spread: mean={df['Crack_Spread'].mean():.2f}, "
          f"min={df['Crack_Spread'].min():.2f}, max={df['Crack_Spread'].max():.2f} $/bbl")


if __name__ == "__main__":
    main()
