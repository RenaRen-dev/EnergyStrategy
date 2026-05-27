"""Continuous front-month crack-spread loader via yfinance.

Provides a single function that returns CL, RB, HO settlement prices
and the 3:2:1 crack spread in $/bbl.  Used by det_signal.py to build
the deterministic SMA signal without needing the full master dataset.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_continuous_crack(
    start: str = "2014-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download CL/RB/HO front-month futures and compute 3:2:1 crack spread.

    Returns
    -------
    DataFrame with columns ``CL, RB, HO, Crack_Spread`` indexed by date.
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
