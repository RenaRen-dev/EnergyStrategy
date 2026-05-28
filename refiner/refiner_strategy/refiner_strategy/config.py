"""Single source of truth for every tunable knob in the refiner strategy.

Centralising constants here prevents silent divergence between modules
and makes parameter sweeps trivial.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
OUTPUTS_DIR: Path = PROJECT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
TICKERS: List[str] = ["VLO", "MPC", "PSX", "DINO", "PBF", "DK", "CVI"]
AB_BASKET: List[str] = TICKERS
AB_WEIGHTS: Dict[str, float] = {
    "VLO": 0.25,
    "MPC": 0.25,
    "PSX": 0.25,
    "DINO": 0.10,
    "PBF": 0.05,
    "DK": 0.05,
    "CVI": 0.05,
}

# ---------------------------------------------------------------------------
# DET signal
# ---------------------------------------------------------------------------
SMA_WINDOW: int = 10
SMA_MIN_PERIODS: int = 5
CONFIRM_DAYS: int = 2

# Fixed-tenor futures: 3rd prompt = third-nearest live contract
TENOR_PROMPT: int = 3

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
TARGET_DAILY_VOL: float = 0.01
VOL_FLOOR: float = 5e-3
VOL_CAP: float = 0.02
DET_SIGNAL_MAG: float = 0.005

Q90_Q10_TO_SIGMA: float = 2.5631  # 2 * 1.2816 for normal q10/q90

OLD_DEADBAND_LOW: float = 0.40
OLD_DEADBAND_HIGH: float = 0.60
OLD_CONVICTION_SLOPE: float = 5.0

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS: float = 10.0

# ---------------------------------------------------------------------------
# Realised volatility
# ---------------------------------------------------------------------------
REALIZED_VOL_WINDOW: int = 20

# ---------------------------------------------------------------------------
# Chronos model
# ---------------------------------------------------------------------------
CHRONOS_MODEL_ID: str = "amazon/chronos-2"
CONTEXT_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Walk-forward optimisation
# ---------------------------------------------------------------------------
WFO_TRAIN_MONTHS: int = 24
WFO_TEST_MONTHS: int = 6
WFO_PURGE_DAYS: int = 5

OOS_TEST_START: str = "2018-04-01"
OOS_TEST_END: str = "2026-12-31"

LORA_BASE_SEED: int = 42

# ---------------------------------------------------------------------------
# Data / statistics
# ---------------------------------------------------------------------------
DATA_START: str = "2014-01-01"
BETA_WINDOW: int = 60
Z_SCORE_WINDOW: int = 256
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Neutral prediction (all quantiles zero, coin-flip p_up)
# ---------------------------------------------------------------------------
NEUTRAL_PRED: Dict[str, float] = {
    "q10": 0.0,
    "q20": 0.0,
    "q30": 0.0,
    "q40": 0.0,
    "q50": 0.0,
    "q60": 0.0,
    "q70": 0.0,
    "q80": 0.0,
    "q90": 0.0,
    "p_up": 0.5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def latest_run_dir() -> Path | None:
    """Return the most-recent outputs/<timestamp>/ directory, or None."""
    if not OUTPUTS_DIR.exists():
        return None
    dirs = sorted(
        [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return dirs[-1] if dirs else None
