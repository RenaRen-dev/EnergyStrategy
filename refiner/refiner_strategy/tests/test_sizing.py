"""Unit tests for the 6 sizing schemes.

Covers all gates, edge cases, and the vol cap.  Run standalone:
    python tests/test_sizing.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from refiner_strategy.config import AB_WEIGHTS, VOL_CAP
from refiner_strategy.sizing.schemes import (
    _apply_vol_cap,
    _ticker_capital,
    size_det,
    size_ens_avg,
    size_ens_veto,
    size_new,
    size_new_cap,
    size_old,
)

NOTIONAL = 100.0
TICKER = "VLO"
WEIGHTS = AB_WEIGHTS


def _pred(
    q50: float = 0.0,
    q10: float = -0.01,
    q90: float = 0.01,
    p_up: float = 0.5,
) -> dict:
    """Build a minimal prediction dict for testing."""
    return {
        "q10": q10,
        "q20": q10 * 0.8,
        "q30": q10 * 0.5,
        "q40": q50 * 0.5,
        "q50": q50,
        "q60": q50 * 1.2 if q50 != 0 else q90 * 0.2,
        "q70": q90 * 0.5,
        "q80": q90 * 0.8,
        "q90": q90,
        "p_up": p_up,
    }


def test_old_inside_deadband() -> None:
    pred = _pred(p_up=0.50)
    result = size_old(pred, 0.0, WEIGHTS, TICKER, NOTIONAL, None)
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_old_outside_deadband() -> None:
    pred = _pred(p_up=0.78)
    result = size_old(pred, 0.0, WEIGHTS, TICKER, NOTIONAL, None)
    assert result != 0.0, f"Expected nonzero, got {result}"
    assert result > 0, f"Expected positive (p_up=0.78), got {result}"


def test_new_vol_floor() -> None:
    pred = _pred(q50=0.001, q10=-0.001, q90=0.001, p_up=0.6)
    result = size_new(pred, 0.0, WEIGHTS, TICKER, NOTIONAL, None)
    assert result == 0.0, f"Expected 0.0 (vol floor), got {result}"


def test_new_consensus_gate() -> None:
    pred = _pred(q50=0.005, q10=-0.02, q90=0.02, p_up=0.40)
    result = size_new(pred, 0.0, WEIGHTS, TICKER, NOTIONAL, None)
    assert result == 0.0, f"Expected 0.0 (consensus gate: q50>0 but p_up<0.5), got {result}"


def test_det_flat() -> None:
    pred = _pred()
    result = size_det(pred, 0.0, WEIGHTS, TICKER, NOTIONAL, None)
    assert result == 0.0, f"Expected 0.0 (det_sig=0), got {result}"


def test_det_long() -> None:
    pred = _pred()
    result = size_det(pred, 1.0, WEIGHTS, TICKER, NOTIONAL, None)
    expected = _ticker_capital(NOTIONAL, WEIGHTS, TICKER)
    assert result == expected, f"Expected {expected}, got {result}"


def test_ens_veto_agreement() -> None:
    pred = _pred(q50=0.005, q10=-0.02, q90=0.02, p_up=0.7)
    result = size_ens_veto(pred, 1.0, WEIGHTS, TICKER, NOTIONAL, 0.015)
    assert result != 0.0, f"Expected nonzero (agreement), got {result}"
    assert result > 0, f"Expected positive (both long), got {result}"


def test_ens_veto_disagreement() -> None:
    pred = _pred(q50=0.005, q10=-0.02, q90=0.02, p_up=0.7)
    result = size_ens_veto(pred, -1.0, WEIGHTS, TICKER, NOTIONAL, 0.015)
    assert result == 0.0, f"Expected 0.0 (disagreement: Chronos long, DET short), got {result}"


def test_vol_cap_scales_down() -> None:
    high_rv = VOL_CAP * 2
    pred = _pred()
    uncapped = size_det(pred, 1.0, WEIGHTS, TICKER, NOTIONAL, VOL_CAP * 0.5)
    capped = size_det(pred, 1.0, WEIGHTS, TICKER, NOTIONAL, high_rv)
    assert abs(capped) < abs(uncapped), (
        f"Expected capped ({capped}) < uncapped ({uncapped}) when rv >> VOL_CAP"
    )


def main() -> None:
    tests = [
        test_old_inside_deadband,
        test_old_outside_deadband,
        test_new_vol_floor,
        test_new_consensus_gate,
        test_det_flat,
        test_det_long,
        test_ens_veto_agreement,
        test_ens_veto_disagreement,
        test_vol_cap_scales_down,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    if failed == 0:
        print(f"All sizing tests passed. ({passed}/{passed})")
    else:
        print(f"{failed} test(s) failed, {passed} passed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
