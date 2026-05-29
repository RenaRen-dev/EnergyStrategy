"""Deterministic crack-spread SMA signal (DET) — README in-house spec.

Follows the Houston Products Desk's rule-based refiner signal
(see ``data/train/README.md`` → "In-house production strategy"), with one
deliberate deviation: the README baseline uses the **M6** (6th-nearby) crack,
but this implementation uses the **M3** (3rd-nearby) crack to match the real
data we actually have (EIA ``RCLC3`` 2015→2024-04 + Databento ``CL.c.2``
recent) and to keep the DET signal consistent with the ``Crack_Z_Score``
Chronos covariate, which is also built from M3.

  1. Per trade_date, build the **M3** 3:2:1 crack spread in $/bbl:
         (2 * RB_M3 + HO_M3) * 42 / 3 − CL_M3
  2. 10-day SMA of the M3 crack.  For the real constant-maturity M3 series this
     is a plain rolling mean; for the per-contract fallback the SMA is grouped
     by the (CL, RB, HO) contract triple so it does not smear across rolls.
  3. Raw signal: +1 when crack > SMA, −1 otherwise (boolean — no flat bucket).
  4. 2-day hold-through confirmation: a flip only takes effect after the new
     sign persists for CONFIRM_DAYS consecutive days; otherwise hold the prior
     position.  The signal is therefore strictly +1 / −1 (never 0).
  5. Position: +1 long basket / −1 short basket, lagged 1 day vs signal — the
     caller must ``.shift(1)`` before trading (MOC entry next day).
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
    load_fixed_tenor_crack_with_contracts,
    load_real_m3_crack,
)


def _grouped_sma(crack: pd.Series, triple: pd.Series) -> pd.Series:
    """10-day SMA of the crack, reset at every (CL, RB, HO) roll.

    Grouping by the active contract triple keeps the rolling mean from
    averaging across a roll date (README step 2).  Because the triple is
    contiguous in time, ``groupby(...).rolling`` is equivalent to restarting
    the window whenever any leg rolls.
    """
    df = pd.DataFrame({"crack": crack.values, "triple": triple.values}, index=crack.index)
    sma = (
        df.groupby("triple", sort=False)["crack"]
        .rolling(SMA_WINDOW, min_periods=SMA_MIN_PERIODS)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return sma.reindex(crack.index)


def _hold_through(raw_sig: pd.Series, n: int) -> pd.Series:
    """N-day hold-through confirmation producing a strictly +1/−1 position.

    Holds the current position until the opposite raw signal has persisted for
    *n* consecutive days, then flips.  Never returns 0 once seeded (README
    step 4).  The position is seeded with the first non-NaN raw signal.
    """
    out = pd.Series(index=raw_sig.index, dtype=float)
    position = 0.0
    pending_sign = 0.0
    pending_count = 0

    for ts, val in raw_sig.items():
        if pd.isna(val):
            out[ts] = position
            continue

        if position == 0.0:
            # Seed: take the first observed sign immediately.
            position = float(val)
            pending_sign = 0.0
            pending_count = 0
        elif val == position:
            # Same side as current position — reset any pending flip.
            pending_sign = 0.0
            pending_count = 0
        else:
            # Opposite side — accumulate consecutive days toward a flip.
            if val == pending_sign:
                pending_count += 1
            else:
                pending_sign = float(val)
                pending_count = 1
            if pending_count >= n:
                position = float(val)
                pending_sign = 0.0
                pending_count = 0

        out[ts] = position

    return out


def _load_crack_and_sma(
    start: str,
    end: str | None,
    tenor_prompt: int,
) -> tuple[pd.Series, pd.Series]:
    """Resolve the crack series and its 10-day SMA from the best source.

    Priority:
      1. Real M3 — EIA (2015→2024-04) stitched with Databento (recent), both
         genuine 3rd-nearby.  Constant-maturity series → plain rolling SMA.
      2. Per-contract Mn hybrid (bundle + yfinance) → SMA grouped by the
         (CL, RB, HO) contract triple so it resets at every roll.
      3. Front-month continuous (last-resort fallback) → plain rolling SMA.
    """
    # 1. Real M3 crack (EIA + Databento, stitched).
    try:
        crack = load_real_m3_crack(start=start, end=end)["Crack_Spread"]
        sma = crack.rolling(SMA_WINDOW, min_periods=SMA_MIN_PERIODS).mean()
        return crack, sma
    except (FileNotFoundError, RuntimeError):
        pass

    # 2. Per-contract hybrid with triple-grouped SMA.
    try:
        crack_df = load_fixed_tenor_crack_with_contracts(
            start=start, end=end, tenor_prompt=tenor_prompt
        )
        crack = crack_df["Crack_Spread"]
        triple = (
            crack_df["CL_contract"].astype(str)
            + "|"
            + crack_df["RB_contract"].astype(str)
            + "|"
            + crack_df["HO_contract"].astype(str)
        )
        return crack, _grouped_sma(crack, triple)
    except RuntimeError as exc:
        print(
            f"[det_signal] per-contract crack unavailable ({exc}); "
            f"falling back to front-month continuous crack spread."
        )

    # 3. Continuous front-month.
    crack = load_continuous_crack(start=start, end=end)["Crack_Spread"]
    sma = crack.rolling(SMA_WINDOW, min_periods=SMA_MIN_PERIODS).mean()
    return crack, sma


def build_det_signal(
    start: str = DATA_START,
    end: str | None = None,
    tenor_prompt: int = TENOR_PROMPT,
) -> pd.DataFrame:
    """Build the README DET signal from the M3 crack.

    Source priority (see ``_load_crack_and_sma``): real M3 (EIA + Databento
    stitched) → per-contract M3 hybrid → continuous front-month.

    Returns a DataFrame with Crack_Spread, SMA10, raw_sig, det_sig.
    ``det_sig`` is strictly +1/−1 and UNLAGGED — caller must ``.shift(1)``.
    """
    crack, sma = _load_crack_and_sma(start, end, tenor_prompt)

    raw_sig = pd.Series(index=crack.index, dtype=float)
    raw_sig[crack > sma] = 1.0
    raw_sig[crack <= sma] = -1.0
    raw_sig[sma.isna()] = float("nan")  # warmup: no SMA yet, leave unseeded

    det_sig = _hold_through(raw_sig, CONFIRM_DAYS)

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
    """Return the full unlagged det_sig Series (+1/−1) up to *test_end*."""
    df = build_det_signal(start=DATA_START, end=test_end, tenor_prompt=tenor_prompt)
    return df["det_sig"]


def diagnose(sig: pd.Series) -> dict:
    """Quick diagnostic counts for a signal Series (n_flat should be ~0)."""
    return {
        "n_long": int((sig == 1).sum()),
        "n_short": int((sig == -1).sum()),
        "n_flat": int((sig == 0).sum()),
        "time_in_market": float((sig != 0).mean()),
    }
