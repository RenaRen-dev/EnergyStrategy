"""Six position-sizing schemes for the refiner strategy.

Each sizer has an identical signature so the A/B harness can call them
interchangeably.  The progression from OLD to ENS_AVG tells the story
of iterative improvement: probability-only (OLD) -> vol-targeted (NEW)
-> vol-capped (NEW_CAP) -> deterministic (DET) -> ensemble (ENS_*).

References:
  - Moskowitz-Ooi-Pedersen 2012: time-series momentum sizing
  - Pedersen 2015 ("Efficiently Inefficient"): vol targeting
  - Bates-Granger 1969: forecast combination (ENS_AVG)
"""
from __future__ import annotations

import math
from typing import Dict

from refiner_strategy.config import (
    DET_SIGNAL_MAG,
    OLD_CONVICTION_SLOPE,
    OLD_DEADBAND_HIGH,
    OLD_DEADBAND_LOW,
    Q90_Q10_TO_SIGMA,
    TARGET_DAILY_VOL,
    VOL_CAP,
    VOL_FLOOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_vol_cap(base: float, realized_vol: float | None) -> float:
    """Scale position down when realised vol exceeds VOL_CAP."""
    if realized_vol is None:
        return base
    try:
        rv = float(realized_vol)
    except (TypeError, ValueError):
        return base
    if rv != rv:  # NaN check (NaN != NaN)
        return base
    if rv <= 0 or rv <= VOL_CAP:
        return base
    return base * (VOL_CAP / rv)


def _ticker_capital(notional: float, weights: Dict[str, float], ticker: str) -> float:
    """Capital allocated to *ticker* within the basket."""
    return notional * weights.get(ticker, 0.0)


# ---------------------------------------------------------------------------
# Sizers
# ---------------------------------------------------------------------------

def size_old(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Original (buggy) probability-only sizer."""
    p_up = pred.get("p_up", 0.5)
    if OLD_DEADBAND_LOW < p_up < OLD_DEADBAND_HIGH:
        return 0.0
    conviction = min(1.0, abs(p_up - 0.5) * OLD_CONVICTION_SLOPE)
    direction = 1.0 if p_up > 0.5 else -1.0
    return direction * conviction * _ticker_capital(notional, weights, ticker)


def size_new(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Vol-targeted sizer with consensus gate."""
    q10 = pred.get("q10", 0.0)
    q50 = pred.get("q50", 0.0)
    q90 = pred.get("q90", 0.0)
    p_up = pred.get("p_up", 0.5)

    raw_forecast_vol = (q90 - q10) / Q90_Q10_TO_SIGMA
    if raw_forecast_vol < VOL_FLOOR:
        return 0.0  # Gate 1: vol floor
    if (q50 > 0) != (p_up > 0.5):
        return 0.0  # Gate 2: consensus

    edge = q50 / raw_forecast_vol
    raw_size = edge / TARGET_DAILY_VOL
    clipped = max(-1.0, min(1.0, raw_size))
    return clipped * _ticker_capital(notional, weights, ticker)


def size_new_cap(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Vol-targeted sizer with realised-vol cap."""
    base = size_new(pred, det_sig, weights, ticker, notional, realized_vol)
    return _apply_vol_cap(base, realized_vol)


def size_det(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Pure deterministic crack-spread signal."""
    if det_sig == 0:
        return 0.0
    base = det_sig * _ticker_capital(notional, weights, ticker)
    return _apply_vol_cap(base, realized_vol)


def size_ens_veto(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Ensemble veto: both Chronos and DET must agree on direction."""
    base = size_new_cap(pred, det_sig, weights, ticker, notional, realized_vol)
    if base == 0.0 or det_sig == 0:
        return 0.0
    if (base > 0) == (det_sig > 0):
        return base
    return 0.0


def size_ens_avg(
    pred: dict,
    det_sig: float,
    weights: Dict[str, float],
    ticker: str,
    notional: float,
    realized_vol: float | None,
) -> float:
    """Bates-Granger 1969 forecast combination: average Chronos q50 with DET."""
    q10 = pred.get("q10", 0.0)
    q50 = pred.get("q50", 0.0)
    q90 = pred.get("q90", 0.0)
    p_up = pred.get("p_up", 0.5)

    raw_forecast_vol = (q90 - q10) / Q90_Q10_TO_SIGMA
    if raw_forecast_vol < VOL_FLOOR:
        return 0.0

    avg_q50 = 0.5 * (q50 + det_sig * DET_SIGNAL_MAG)

    # consensus gate: avg_q50 vs p_up (NOT chronos q50 vs p_up)
    if (avg_q50 > 0) != (p_up > 0.5):
        return 0.0

    edge = avg_q50 / raw_forecast_vol
    raw_size = edge / TARGET_DAILY_VOL
    clipped = max(-1.0, min(1.0, raw_size))
    base = clipped * _ticker_capital(notional, weights, ticker)
    return _apply_vol_cap(base, realized_vol)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIZERS: Dict[str, callable] = {
    "OLD": size_old,
    "NEW": size_new,
    "NEW_CAP": size_new_cap,
    "DET": size_det,
    "ENS_VETO": size_ens_veto,
    "ENS_AVG": size_ens_avg,
}

DEFAULT_SCHEMES = ("OLD", "NEW", "NEW_CAP", "DET", "ENS_VETO", "ENS_AVG")
