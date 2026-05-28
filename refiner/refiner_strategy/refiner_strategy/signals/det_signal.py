"""Deterministic crack-spread SMA signal (DET).

Replicates the Houston Products Desk's rule-based refiner signal:
  1. Compute 3:2:1 crack spread in $/bbl
  2. 10-day SMA
  3. +1 when crack > SMA, -1 when crack < SMA
  4. 2-day persistence confirmation filter
  5. Caller must .shift(1) before trading — the signal is unlagged here

The persistence filter prevents whipsaws: a crossover must hold for
CONFIRM_DAYS consecutive days before the signal flips.
"""
from __future__ import annotations

import pandas as pd

from refiner_strategy.config import (
    CONFIRM_DAYS,
    DATA_START,
    SMA_MIN_PERIODS,
    SMA_WINDOW,
    TENOR_PROMPT,
)
from refiner_strategy.data.futures_loader import (
    load_continuous_crack,
    load_fixed_tenor_crack,
)


def _confirm(raw_sig: pd.Series, n: int) -> pd.Series:
    """N-day persistence filter: signal must equal itself for *n* days."""
    confirmed = raw_sig.copy()
    for i in range(1, n):
        confirmed = confirmed.where(raw_sig == raw_sig.shift(i), 0)
    return confirmed


def build_det_signal(
    start: str = DATA_START,
    end: str | None = None,
    tenor_prompt: int = TENOR_PROMPT,
) -> pd.DataFrame:
    """Build DET signal from live yfinance data.

    Uses fixed-tenor (Nth-prompt) crack spread by default to avoid the
    roll-day discontinuities of the front-month series.  Falls back to
    front-month if Yahoo Finance has insufficient history for the
    requested tenor.

    Returns DataFrame with Crack_Spread, SMA10, raw_sig, det_sig.
    The det_sig is UNLAGGED — caller must .shift(1) before use.
    """
    try:
        crack_df = load_fixed_tenor_crack(start=start, end=end, tenor_prompt=tenor_prompt)
    except RuntimeError as exc:
        print(
            f"[det_signal] Fixed-tenor (prompt={tenor_prompt}) unavailable "
            f"({exc}); falling back to front-month continuous crack spread."
        )
        crack_df = load_continuous_crack(start=start, end=end)
    crack = crack_df["Crack_Spread"]

    sma = crack.rolling(SMA_WINDOW, min_periods=SMA_MIN_PERIODS).mean()

    raw_sig = pd.Series(0, index=crack.index, dtype=float)
    raw_sig[crack > sma] = 1.0
    raw_sig[crack < sma] = -1.0

    det_sig = _confirm(raw_sig, CONFIRM_DAYS)

    return pd.DataFrame(
        {
            "Crack_Spread": crack,
            "SMA10": sma,
            "raw_sig": raw_sig,
            "det_sig": det_sig,
        }
    )


def build_stitched_det_signal(
    test_end: str | None = None,
    tenor_prompt: int = TENOR_PROMPT,
) -> pd.Series:
    """Return the full unlagged det_sig Series up to *test_end*."""
    df = build_det_signal(start=DATA_START, end=test_end, tenor_prompt=tenor_prompt)
    return df["det_sig"]


def diagnose(sig: pd.Series) -> dict:
    """Quick diagnostic counts for a signal Series."""
    return {
        "n_long": int((sig == 1).sum()),
        "n_short": int((sig == -1).sum()),
        "n_flat": int((sig == 0).sum()),
        "time_in_market": float((sig != 0).mean()),
    }
