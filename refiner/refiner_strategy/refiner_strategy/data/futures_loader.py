"""Fixed-tenor crack-spread loader.

Loads CL / RB / HO futures at a **fixed tenor** (default = 3rd prompt)
and computes the 3:2:1 crack spread.  This avoids the roll-day
discontinuities present in front-month continuous series.

Why fixed tenor?
----------------
Front-month series (e.g. ``CL=F``) automatically roll into the next
contract when the current one expires.  That roll creates an artificial
price jump (the new contract usually trades at a different level than
the expiring one).  The DET signal is a 10-day SMA crossover; a single
~3% roll gap can trigger a spurious crossover and a false trade.

Fixed-tenor rule: every calendar day, use the **Nth-prompt** contract.
The 1st prompt is the nearest live contract, the 2nd prompt the next,
and so on.  Picking the 3rd prompt keeps us roughly two months away
from expiry — far enough from the noisy delivery window, close enough
to remain liquid.

Data sources
------------
1. **Local CSVs** in ``<project_root>/data/futures/{PRODUCT}_{YYYY}_{MM}.csv``
   (one file per delivery month).  Schema:
   ``date, settlement, expiry_date, days_to_expiry, open_interest, total_volume``
   The expiry comes directly from the file — no approximation needed.

2. **yfinance** as a fill-in for delivery months *after* the CSVs end
   (i.e., contracts that didn't exist when the CSV snapshot was taken).

CSV contracts always take precedence when both sources cover the same
delivery month.

Module entry points
-------------------
- ``load_fixed_tenor_crack(start, end, tenor_prompt=3)`` — the new default.
- ``load_continuous_crack(start, end)`` — legacy front-month, kept as a
  documented fallback for cases where the CSV chain plus yfinance still
  cannot provide enough Nth-prompt coverage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# Directory holding per-delivery-month CSV files: data/futures/{PROD}_{YYYY}_{MM}.csv
# Path layout:
#   <project_root>/refiner_strategy/data/futures_loader.py   ← this file
#   <project_root>/data/futures/                              ← CSV directory
FUTURES_CSV_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "futures"

# Yahoo Finance / CME futures month codes
_MONTH_CODES: Dict[int, str] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

# Conservative expiry-day approximations (day of preceding month).
# Only used for *yfinance* contracts where the CSV's exact expiry is missing.
_EXPIRY_DAY: Dict[str, int] = {
    "CL": 20,
    "RB": 25,
    "HO": 25,
}


# ---------------------------------------------------------------------------
# Local-CSV loader
# ---------------------------------------------------------------------------

def _parse_csv_filename(path: Path) -> tuple[str, int, int] | None:
    """Parse ``CL_2018_06.csv`` → ``("CL", 2018, 6)``; return None if malformed."""
    parts = path.stem.split("_")
    if len(parts) != 3:
        return None
    product, year_str, month_str = parts
    try:
        return product, int(year_str), int(month_str)
    except ValueError:
        return None


def _load_local_chain(product: str, csv_dir: Path = FUTURES_CSV_DIR) -> List[dict]:
    """Load every local-CSV contract for ``product`` (e.g. ``CL``).

    Each contract dict has keys ``symbol, delivery, expiry, close`` where
    ``close`` is a pd.Series of settlement prices indexed by trading date.
    """
    if not csv_dir.is_dir():
        return []

    contracts: List[dict] = []
    for f in sorted(csv_dir.glob(f"{product}_*.csv")):
        parsed = _parse_csv_filename(f)
        if parsed is None or parsed[0] != product:
            continue
        _, year, month = parsed

        try:
            df = pd.read_csv(f, parse_dates=["date", "expiry_date"])
        except (ValueError, KeyError):
            continue
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

        # Expiry from the data itself (more accurate than the approximation).
        expiry = pd.Timestamp(df["expiry_date"].iloc[0])

        contracts.append(
            {
                "symbol": f.stem,
                "delivery": pd.Timestamp(year, month, 1),
                "expiry": expiry,
                "close": close,
            }
        )
    return contracts


# ---------------------------------------------------------------------------
# yfinance fill-in
# ---------------------------------------------------------------------------

def _contract_symbol(product: str, year: int, month: int) -> str:
    """Yahoo Finance ticker for a specific delivery contract (e.g. ``CLM24.NYM``)."""
    return f"{product}{_MONTH_CODES[month]}{str(year)[-2:]}.NYM"


def _contract_expiry_approx(product: str, year: int, month: int) -> pd.Timestamp:
    """Approximate expiry: a day in the month *preceding* delivery."""
    day = _EXPIRY_DAY[product]
    if month == 1:
        return pd.Timestamp(year - 1, 12, day)
    return pd.Timestamp(year, month - 1, day)


def _strip_tz(s: pd.Series) -> pd.Series:
    """Drop timezone info from a DatetimeIndex so series can be aligned by date."""
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _batch_fetch_yfinance(
    contracts_info: List[dict],
    start: str,
    end: str | None,
) -> List[dict]:
    """Batch-download yfinance contracts; attach a ``close`` series to each dict."""
    if not contracts_info:
        return []
    symbols = [c["symbol"] for c in contracts_info]
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        return []

    fetched: List[dict] = []
    if len(symbols) == 1:
        sym = symbols[0]
        if "Close" in raw.columns:
            s = raw["Close"].dropna()
            if not s.empty:
                fetched.append({**contracts_info[0], "close": _strip_tz(s)})
        return fetched

    for info in contracts_info:
        try:
            s = raw[info["symbol"]]["Close"].dropna()
            if not s.empty:
                fetched.append({**info, "close": _strip_tz(s)})
        except (KeyError, AttributeError):
            continue
    return fetched


def _enumerate_yfinance_contracts(
    product: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exclude_deliveries: set,
) -> List[dict]:
    """List yfinance contract info for delivery months not already in ``exclude_deliveries``."""
    out: List[dict] = []
    for year in range(start.year - 1, end.year + 2):
        for month in range(1, 13):
            delivery = pd.Timestamp(year, month, 1)
            if delivery in exclude_deliveries:
                continue
            expiry = _contract_expiry_approx(product, year, month)
            # Skip contracts that expired before the window or are too far out.
            if expiry < start - pd.DateOffset(months=1):
                continue
            if expiry > end + pd.DateOffset(months=24):
                continue
            out.append(
                {
                    "symbol": _contract_symbol(product, year, month),
                    "delivery": delivery,
                    "expiry": expiry,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Nth-prompt series construction
# ---------------------------------------------------------------------------

def _build_nth_prompt_series(
    product: str,
    start: str,
    end: str | None,
    tenor_prompt: int,
    csv_dir: Path = FUTURES_CSV_DIR,
) -> pd.Series:
    """Build a fixed-tenor (Nth-prompt) close-price series for one product."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    # 1. Load every CSV contract for this product.
    local_contracts = _load_local_chain(product, csv_dir=csv_dir)
    local_deliveries = {c["delivery"] for c in local_contracts}

    # 2. Ask yfinance for delivery months that CSVs do not cover and whose
    #    life intersects [start, end].
    yfinance_info = _enumerate_yfinance_contracts(
        product, start_ts, end_ts, exclude_deliveries=local_deliveries
    )
    yfinance_contracts = _batch_fetch_yfinance(yfinance_info, start, end)

    contracts = local_contracts + yfinance_contracts
    if not contracts:
        raise RuntimeError(
            f"No {product} contracts found.  Checked local CSVs in "
            f"{csv_dir} and yfinance — both empty."
        )

    # 3. For each trading day in [start, end], pick the Nth-prompt live contract.
    all_dates = sorted({d for c in contracts for d in c["close"].index})
    all_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    rows: Dict[pd.Timestamp, float] = {}
    for d in all_dates:
        live = [c for c in contracts if c["expiry"] > d and d in c["close"].index]
        live.sort(key=lambda c: c["expiry"])
        if len(live) >= tenor_prompt:
            rows[d] = float(live[tenor_prompt - 1]["close"].loc[d])

    return pd.Series(rows, name=product).sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_fixed_tenor_crack(
    start: str = "2014-01-01",
    end: str | None = None,
    tenor_prompt: int = 3,
    csv_dir: Path = FUTURES_CSV_DIR,
) -> pd.DataFrame:
    """Load CL/RB/HO at a fixed tenor and compute the 3:2:1 crack spread.

    Parameters
    ----------
    start, end : ISO-formatted date strings (``end`` is exclusive).
    tenor_prompt : 1 = front-month, 3 = 3rd prompt (default), etc.
    csv_dir : directory containing per-delivery-month CSV files.

    Returns
    -------
    DataFrame with columns ``CL, RB, HO, Crack_Spread`` indexed by date.
    """
    cl = _build_nth_prompt_series("CL", start, end, tenor_prompt, csv_dir=csv_dir)
    rb = _build_nth_prompt_series("RB", start, end, tenor_prompt, csv_dir=csv_dir)
    ho = _build_nth_prompt_series("HO", start, end, tenor_prompt, csv_dir=csv_dir)

    df = pd.DataFrame({"CL": cl, "RB": rb, "HO": ho}).dropna()
    if df.empty:
        raise RuntimeError(
            "Empty crack-spread DataFrame after fixed-tenor alignment.  "
            "Most likely CL/RB/HO have no overlapping dates at the requested tenor."
        )

    df["Crack_Spread"] = (2 * df["RB"] * 42 + df["HO"] * 42) / 3 - df["CL"]
    return df


def load_continuous_crack(
    start: str = "2014-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Legacy front-month loader (deprecated — kept for fallback).

    Uses Yahoo's auto-rolling continuous tickers (``CL=F``, ``RB=F``,
    ``HO=F``).  Suffers from roll-day price gaps that can fire spurious
    DET signals.  Prefer ``load_fixed_tenor_crack(tenor_prompt=3)``.
    """
    tickers = ["CL=F", "RB=F", "HO=F"]
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for futures tickers")

    # yfinance sorts multi-ticker columns alphabetically:
    # ["CL=F", "RB=F", "HO=F"] -> [CL=F, HO=F, RB=F]
    close = raw["Close"].copy()
    close.columns = ["CL", "HO", "RB"]

    close = close.dropna()
    if close.empty:
        raise RuntimeError("All-NaN futures prices after dropna")

    close["Crack_Spread"] = (2 * close["RB"] * 42 + close["HO"] * 42) / 3 - close["CL"]
    return close
