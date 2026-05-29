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

# Directories holding per-delivery-month CSV files:
#   data/cache/futures/    — yfinance downloads (user-managed cache)
#   data/train/futures/    — README bundle (≤2021-12-31)
# Path layout:
#   <project_root>/refiner_strategy/data/futures_loader.py   ← this file
#   <project_root>/data/cache/futures/                       ← cache (checked first)
#   <project_root>/data/train/futures/                       ← bundle (fallback)
FUTURES_CACHE_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "cache" / "futures"
FUTURES_CSV_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "train" / "futures"
)

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

# A genuine Nth-prompt curve must include the *near* part of the term structure.
# If the nearest live contract on a date expires more than this many days out, we
# only have far-dated contracts (e.g. yfinance after the local bundle ends) and
# CANNOT form a valid M6 — those dates are skipped so the caller falls back to the
# continuous front-month series instead of fabricating a wrong "6th-nearby".
_MAX_FRONT_DAYS_TO_EXPIRY: int = 60

# Cache files with fewer rows than this are treated as empty (yfinance returns a
# stale 1-row stub for delisted/expired contracts) and skipped.
_MIN_CONTRACT_ROWS: int = 20


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

    Checks cache dir first (user downloads), then falls back to bundle dir.
    Each contract dict has keys ``symbol, delivery, expiry, close`` where
    ``close`` is a pd.Series of settlement prices indexed by trading date.
    """
    contracts: List[dict] = []

    # Try cache dir first (user downloads)
    for search_dir in [FUTURES_CACHE_DIR, csv_dir]:
        if not search_dir.is_dir():
            continue

        for f in sorted(search_dir.glob(f"{product}_*.csv")):
            parsed = _parse_csv_filename(f)
            if parsed is None or parsed[0] != product:
                continue
            _, year, month = parsed

            # Check if we already loaded this contract from cache
            if any(c["delivery"] == pd.Timestamp(year, month, 1) for c in contracts):
                continue

            # Try to load as bundle format (with expiry_date column)
            try:
                df = pd.read_csv(f, parse_dates=["date", "expiry_date"])
                if "settlement" in df.columns:
                    close = (
                        df.set_index("date")["settlement"]
                        .dropna()
                        .astype(float)
                        .sort_index()
                    )
                    expiry = pd.Timestamp(df["expiry_date"].iloc[0])
                else:
                    continue
            except (ValueError, KeyError, TypeError):
                # Fallback: cache format (date + Close, optional expiry_date).
                try:
                    df = pd.read_csv(f, parse_dates=["date"])
                except (ValueError, KeyError, TypeError):
                    continue
                price_col = "Close" if "Close" in df.columns else (
                    "settlement" if "settlement" in df.columns else None
                )
                if price_col is None:
                    continue
                close = (
                    df.set_index("date")[price_col]
                    .dropna()
                    .astype(float)
                    .sort_index()
                )
                # Expiry: prefer a stored column, else derive from the delivery
                # month encoded in the filename (NOT last_date — that would be
                # "today" for a still-trading contract and corrupt M6 ordering).
                if "expiry_date" in df.columns and df["expiry_date"].notna().any():
                    expiry = pd.Timestamp(df["expiry_date"].dropna().iloc[0])
                else:
                    expiry = _contract_expiry_approx(product, year, month)

            # Skip empty / stale-stub files (yfinance returns a 1-row stub for
            # delisted contracts).
            if len(close) < _MIN_CONTRACT_ROWS:
                continue

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
        if len(live) < tenor_prompt:
            continue
        # Guard: the front contract must be genuinely near-dated, else we only
        # have far-dated contracts and cannot form a valid Nth-prompt (skip date).
        if (live[0]["expiry"] - d).days > _MAX_FRONT_DAYS_TO_EXPIRY:
            continue
        rows[d] = float(live[tenor_prompt - 1]["close"].loc[d])

    return pd.Series(rows, name=product).sort_index()


def _build_nth_prompt_with_contract(
    product: str,
    start: str,
    end: str | None,
    tenor_prompt: int,
    csv_dir: Path = FUTURES_CSV_DIR,
) -> pd.DataFrame:
    """Like ``_build_nth_prompt_series`` but also records the contract identity.

    Returns a DataFrame indexed by date with columns ``<product>`` (settlement
    of the Nth-prompt contract) and ``<product>_contract`` (the delivery-month
    identifier of that contract, e.g. ``CL_2020_07``).  The contract id lets the
    DET signal group its rolling SMA by the active (CL, RB, HO) triple so the
    mean never smears across a roll date (README step 2).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    local_contracts = _load_local_chain(product, csv_dir=csv_dir)
    local_deliveries = {c["delivery"] for c in local_contracts}

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

    all_dates = sorted({d for c in contracts for d in c["close"].index})
    all_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    price_rows: Dict[pd.Timestamp, float] = {}
    contract_rows: Dict[pd.Timestamp, str] = {}
    for d in all_dates:
        live = [c for c in contracts if c["expiry"] > d and d in c["close"].index]
        live.sort(key=lambda c: c["expiry"])
        if len(live) < tenor_prompt:
            continue
        # Guard: front contract must be near-dated (see _MAX_FRONT_DAYS_TO_EXPIRY).
        if (live[0]["expiry"] - d).days > _MAX_FRONT_DAYS_TO_EXPIRY:
            continue
        pick = live[tenor_prompt - 1]
        price_rows[d] = float(pick["close"].loc[d])
        contract_rows[d] = str(pick["symbol"])

    df = pd.DataFrame(
        {
            product: pd.Series(price_rows),
            f"{product}_contract": pd.Series(contract_rows),
        }
    ).sort_index()
    return df


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


def load_fixed_tenor_crack_with_contracts(
    start: str = "2015-01-01",
    end: str | None = None,
    tenor_prompt: int = 6,
    csv_dir: Path = FUTURES_CSV_DIR,
) -> pd.DataFrame:
    """Load CL/RB/HO at a fixed tenor with per-leg contract identity.

    Returns a DataFrame indexed by date with columns
    ``CL, RB, HO, Crack_Spread, CL_contract, RB_contract, HO_contract``.

    The ``*_contract`` columns identify which physical delivery month is the
    Nth-prompt pick on each date, so the DET signal can compute its 10-day SMA
    *grouped by the (CL, RB, HO) triple* — the mean then resets cleanly at every
    roll instead of smearing across it (README "In-house production strategy",
    step 2).

    yfinance only carries currently-listed contracts, so true M6 is available for
    the bundle period (≤2021) and the recent live window, but NOT for the gap in
    between (expired contracts).  Any gap is filled with the continuous
    front-month series (level-aligned to the M6 series and labelled ``CONTINUOUS``
    so the SMA still resets at the M6↔continuous boundary).  The returned series
    is therefore gap-free over [start, end].
    """
    cl = _build_nth_prompt_with_contract("CL", start, end, tenor_prompt, csv_dir=csv_dir)
    rb = _build_nth_prompt_with_contract("RB", start, end, tenor_prompt, csv_dir=csv_dir)
    ho = _build_nth_prompt_with_contract("HO", start, end, tenor_prompt, csv_dir=csv_dir)

    df = cl.join([rb, ho], how="inner").dropna(subset=["CL", "RB", "HO"])
    if not df.empty:
        df["Crack_Spread"] = (2 * df["RB"] * 42 + df["HO"] * 42) / 3 - df["CL"]

    return _fill_gaps_with_continuous(df, start, end)


def _fill_gaps_with_continuous(
    m6: pd.DataFrame,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Fill missing trading days in the M6 crack with continuous front-month.

    The continuous crack is shifted by the median (M6 − continuous) offset over
    overlapping dates so the two segments line up in level.  Filled rows are
    labelled with ``*_contract = "CONTINUOUS"`` so downstream SMA grouping treats
    the gap as one block (and resets at the boundary with the real-M6 segment).
    """
    continuous = load_continuous_crack(start=start, end=end)  # cache-aware
    if continuous.empty:
        if m6.empty:
            raise RuntimeError(
                "No M6 contracts and no continuous front-month available."
            )
        return m6

    cont_crack = continuous["Crack_Spread"]

    if m6.empty:
        missing_idx = cont_crack.index
        offset = 0.0
    else:
        missing_idx = cont_crack.index.difference(m6.index)
        common = m6.index.intersection(cont_crack.index)
        offset = (
            float((m6["Crack_Spread"].loc[common] - cont_crack.loc[common]).median())
            if len(common) > 0 else 0.0
        )

    if len(missing_idx) == 0:
        return m6.sort_index()

    fill = pd.DataFrame(index=missing_idx)
    fill["CL"] = continuous["CL"].loc[missing_idx]
    fill["HO"] = continuous["HO"].loc[missing_idx]
    fill["RB"] = continuous["RB"].loc[missing_idx]
    fill["Crack_Spread"] = cont_crack.loc[missing_idx] + offset
    fill["CL_contract"] = "CONTINUOUS"
    fill["RB_contract"] = "CONTINUOUS"
    fill["HO_contract"] = "CONTINUOUS"

    combined = pd.concat([m6, fill]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# Cached continuous front-month, written by scripts/00_download_and_cache_data.py
CONTINUOUS_CACHE_FILE: Path = (
    Path(__file__).resolve().parents[2] / "data" / "cache" / "continuous_front_month.csv"
)

# Pre-built M3 crack files (date, CL, RB, HO[, Crack_Spread]) written by the
# download scripts.  Databento covers the full period from one real source;
# EIA is real but ends April 2024 (series discontinued).
DATABENTO_CRACK_FILE: Path = (
    Path(__file__).resolve().parents[2] / "data" / "databento" / "futures_m3.csv"
)
EIA_CRACK_FILE: Path = (
    Path(__file__).resolve().parents[2] / "data" / "eia" / "futures_m3.csv"
)


def _read_crack_csv(path: Path, start: str, end: str | None, source: str) -> pd.DataFrame:
    """Read a pre-built CL/RB/HO crack CSV, compute the spread, slice to window."""
    if not path.is_file():
        raise FileNotFoundError(f"{source} crack file not found at {path}.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Crack_Spread" not in df.columns:
        df["Crack_Spread"] = (2 * df["RB"] * 42 + df["HO"] * 42) / 3 - df["CL"]
    df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise RuntimeError(f"{source} crack file has no rows in the requested window.")
    return df


def load_databento_crack(
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Load the real M3 crack downloaded from Databento (data/databento/futures_m3.csv).

    Returns a DataFrame indexed by date with columns ``CL, RB, HO, Crack_Spread``.
    Sourced from CME Globex via continuous symbology (CL.c.2/RB.c.2/HO.c.2), this
    is genuine 3rd-nearby data covering the full period from one consistent source.
    Raises FileNotFoundError if the Databento download hasn't run.
    """
    return _read_crack_csv(DATABENTO_CRACK_FILE, start, end, "Databento")


def load_eia_crack(
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Load the real M3 crack downloaded from EIA (data/eia/futures_m3.csv).

    Genuine 3rd-nearby data, but EIA discontinued these series in April 2024, so
    the file ends ~2024-04-05.  Combine with Databento for full coverage.
    Raises FileNotFoundError if the EIA download hasn't run.
    """
    return _read_crack_csv(EIA_CRACK_FILE, start, end, "EIA")


def load_real_m3_crack(
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Real M3 crack stitched from the available real sources (EIA + Databento).

    Both EIA (``RCLC3`` etc.) and Databento (``CL.c.2`` etc.) are genuine
    3rd-nearby series.  EIA covers 2015→2024-04 (then discontinued); Databento
    covers the recent period.  This returns a single real-M3 series:

      * if Databento already covers the requested start, use it alone;
      * else use EIA for its range and append the Databento tail beyond EIA's
        last date, shifted by the (EIA − Databento) offset at the last
        overlapping date so the join is seamless.

    Raises FileNotFoundError if neither source has been downloaded.
    """
    eia = dbo = None
    try:
        eia = load_eia_crack(start=start, end=end)
    except (FileNotFoundError, RuntimeError):
        pass
    try:
        dbo = load_databento_crack(start=start, end=end)
    except (FileNotFoundError, RuntimeError):
        pass

    if eia is None and dbo is None:
        raise FileNotFoundError(
            "No real M3 crack found. Run scripts/00_download_eia_futures.py "
            "and/or scripts/00_download_databento_futures.py first."
        )
    if eia is None:
        return dbo
    if dbo is None:
        return eia

    # Databento already spans the requested start → use it alone (one source).
    if dbo.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=31):
        return dbo

    # Otherwise: EIA for its range, Databento appended beyond EIA's end.
    boundary = eia.index.max()
    common = eia.index.intersection(dbo.index)
    offset = (
        float(eia["Crack_Spread"].loc[common.max()] - dbo["Crack_Spread"].loc[common.max()])
        if len(common) > 0 else 0.0
    )
    dbo_tail = dbo.loc[dbo.index > boundary].copy()
    if not dbo_tail.empty:
        dbo_tail["Crack_Spread"] = dbo_tail["Crack_Spread"] + offset
    combined = pd.concat([eia, dbo_tail]).sort_index()
    return combined[~combined.index.duplicated(keep="first")]


def load_continuous_crack(
    start: str = "2014-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Front-month crack loader (fallback when per-contract M6 is unavailable).

    Reads the cached continuous front-month (CL/HO/RB) if present, else downloads
    Yahoo's auto-rolling continuous tickers (``CL=F``, ``RB=F``, ``HO=F``).
    Suffers from roll-day price gaps; used only where per-contract M6 cannot be
    reconstructed (e.g. the 2022+ test period where expired contracts are gone).
    """
    close: pd.DataFrame | None = None

    # 1. Local cache first (offline / reproducible).
    if CONTINUOUS_CACHE_FILE.is_file():
        cached = pd.read_csv(CONTINUOUS_CACHE_FILE, index_col=0, parse_dates=True)
        if {"CL", "HO", "RB"}.issubset(cached.columns):
            close = cached[["CL", "HO", "RB"]].copy()

    # 2. Else live download.
    if close is None:
        tickers = ["CL=F", "RB=F", "HO=F"]
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
        if raw.empty:
            raise RuntimeError("yfinance returned empty data for futures tickers")
        # yfinance sorts multi-ticker columns alphabetically: [CL=F, HO=F, RB=F]
        close = raw["Close"].copy()
        close.columns = ["CL", "HO", "RB"]

    # Restrict to requested window.
    start_ts = pd.Timestamp(start)
    close = close.loc[close.index >= start_ts]
    if end is not None:
        close = close.loc[close.index <= pd.Timestamp(end)]

    close = close.dropna()
    if close.empty:
        raise RuntimeError("All-NaN futures prices after dropna")

    close["Crack_Spread"] = (2 * close["RB"] * 42 + close["HO"] * 42) / 3 - close["CL"]
    return close
