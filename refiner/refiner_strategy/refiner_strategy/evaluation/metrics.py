"""Performance metrics: Sharpe, drawdown, hit rate.

All metrics are computed from a daily PnL series in dollar terms.
The hit-rate function uses effective_size (H4 fix) to correctly
attribute returns to the position that was actually held.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from refiner_strategy.config import TRADING_DAYS_PER_YEAR


def metrics_from_pnl(pnl: pd.Series, notional: float = 100.0) -> dict:
    """Compute standard performance metrics from a daily PnL series.

    Parameters
    ----------
    pnl : daily PnL in dollar terms
    notional : strategy notional for converting to returns
    """
    if pnl.empty or len(pnl) < 2:
        return {
            "cum_pnl": 0.0,
            "ann_ret": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "calmar": 0.0,
            "n": 0,
        }

    daily_ret = pnl / notional
    cum_pnl = float(pnl.sum())

    mu = daily_ret.mean()
    sigma = daily_ret.std()
    ann_ret = float(mu * TRADING_DAYS_PER_YEAR)
    sharpe = float(math.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    sortino = (
        float(math.sqrt(TRADING_DAYS_PER_YEAR) * mu / downside_std)
        if downside_std > 0
        else 0.0
    )

    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd_pct = float(drawdown.min())

    calmar = float(ann_ret / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0

    return {
        "cum_pnl": cum_pnl,
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd_pct,
        "calmar": calmar,
        "n": len(pnl),
    }


def horizon_aligned_hit_rate(trades_df: pd.DataFrame) -> float:
    """Fraction of active days where held position earned positive PnL.

    H4 FIX: uses effective_size (what was HELD during T) not
    target_size (what was DECIDED at T-1 close).
    """
    if trades_df.empty or "effective_size" not in trades_df.columns:
        return float("nan")

    active = trades_df[trades_df["effective_size"] != 0]
    if active.empty:
        return float("nan")

    hits = (active["effective_size"] * active["actual_ret"] > 0).sum()
    return float(hits / len(active))
