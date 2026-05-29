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

# Bundled in-house data (README data export): used for ≤2021 history.
# Recent (2022+) data is pulled from yfinance.
LOCAL_DATA_DIR: Path = PROJECT_DIR / "data" / "train"

# EIA-sourced futures (real per-contract M1-M4 settlements, free API).
# Downloaded by scripts/00_download_eia_futures.py into its own folder.
# NOTE: EIA discontinued these daily series in April 2024.
EIA_DATA_DIR: Path = PROJECT_DIR / "data" / "eia"

# Databento-sourced futures (real CME M3 via continuous symbology, full period).
# Downloaded by scripts/00_download_databento_futures.py into its own folder.
DATABENTO_DATA_DIR: Path = PROJECT_DIR / "data" / "databento"

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

# Fixed-tenor futures: M3 = 3rd-nearby contract.
# EIA provides real per-contract settlements only for contracts 1-4, so M3 is
# the chosen tenor (deep enough to avoid the noisy expiry window, fully covered
# by the free EIA source across the entire 2015-2025 period).
TENOR_PROMPT: int = 3

# EIA NYMEX-futures series IDs for the M3 (3rd-nearby) crack legs.
# Verify at https://www.eia.gov/opendata/browser/petroleum/pri/fut
EIA_SERIES_M3: Dict[str, str] = {
    "CL": "RCLC3",                     # Cushing OK WTI crude, contract 3 ($/bbl)
    "RB": "EER_EPMRR_PE3_Y35NY_DPG",   # NY Harbor RBOB gasoline, contract 3 ($/gal)
    "HO": "EER_EPD2F_PE3_Y35NY_DPG",   # NY Harbor No.2 heating oil, contract 3 ($/gal)
}

# Databento CME Globex dataset + continuous symbols for the M3 crack legs.
# Continuous symbology: {ROOT}.c.{rank}, rank 2 = 3rd-nearby (calendar roll).
# RB/HO settle in $/gal (×42 → $/bbl); CL in $/bbl.
DATABENTO_DATASET: str = "GLBX.MDP3"
DATABENTO_SYMBOLS_M3: Dict[str, str] = {
    "CL": "CL.c.2",   # WTI crude, 3rd-nearby ($/bbl)
    "RB": "RB.c.2",   # RBOB gasoline, 3rd-nearby ($/gal)
    "HO": "HO.c.2",   # NY Harbor ULSD / heating oil, 3rd-nearby ($/gal)
}

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

# Annual borrow cost charged on short exposure (long/short simulator default).
BORROW_COST_BPS_PER_YEAR: float = 50.0

# README reporting notional: the in-house headline P&L is quoted on a constant
# $10M notional ("Total $ PnL on $10M notional").  Daily $ PnL = daily return ×
# this notional (additive), so total/per-year $ PnL is comparable to the README.
README_NOTIONAL: float = 10_000_000.0

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
WFO_TRAIN_MONTHS: int = 24   # 2-year rolling train
WFO_TEST_MONTHS: int = 6     # 6-month test
WFO_PURGE_DAYS: int = 5

OOS_TEST_START: str = "2018-04-01"   # folds begin here (gives ~17 folds to data-end)
OOS_TEST_END: str = "2026-12-31"

LORA_BASE_SEED: int = 42

# ---------------------------------------------------------------------------
# Data / statistics
# ---------------------------------------------------------------------------
# README bundle: train ≤ 2021-12-31, test ≥ 2022-01-01. History starts 2015.
DATA_START: str = "2015-01-01"
TEST_START: str = "2022-01-01"
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
