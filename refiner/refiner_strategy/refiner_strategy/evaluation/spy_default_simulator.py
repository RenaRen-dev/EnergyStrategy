"""SPY-default overlay simulator.

When the refiner strategy has no position, the capital sits in SPY
(the "default" allocation).  This module answers: "What if we ran
the refiner strategy as a tactical overlay on top of a passive SPY
portfolio?"

The key insight is that unused capital earns the market return instead
of sitting idle, which is a more realistic comparison than standalone
refiner returns vs SPY buy-and-hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from refiner_strategy.config import AB_WEIGHTS, TICKERS, TRADING_DAYS_PER_YEAR
from refiner_strategy.evaluation.metrics import metrics_from_pnl


@dataclass
class SpyDefaultConfig:
    """Cost assumptions for the SPY-default simulation."""
    txn_cost_bps_per_leg: float = 5.0
    borrow_cost_bps_per_year: float = 0.0
    leverage_cap: float = 1.0


def fetch_unhedged_returns(test_end: str | None = None) -> pd.Series:
    """Download TICKERS prices from yfinance and return cap-weighted basket returns."""
    raw = yf.download(TICKERS, start="2014-01-01", end=test_end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for equity tickers")
    close = raw["Close"]
    rets = close.pct_change().dropna()

    basket = sum(rets[t] * AB_WEIGHTS[t] for t in TICKERS)
    return basket


def simulate_spy_default(
    spy_returns: pd.Series,
    refiner_returns: pd.Series,
    trades_df: pd.DataFrame,
    config: SpyDefaultConfig | None = None,
    strategy_notional: float = 100.0,
) -> dict:
    """Simulate a SPY-default portfolio with refiner overlay.

    Returns dict with nav_series and metrics.
    """
    if config is None:
        config = SpyDefaultConfig()

    # Daily allocation = sum of target_size / notional
    alloc_by_date = trades_df.groupby("date")["target_size"].sum() / strategy_notional

    common_idx = spy_returns.index.intersection(alloc_by_date.index)
    if common_idx.empty:
        return {"nav_series": pd.Series(dtype=float), "metrics": {}}

    nav = [1.0]
    prev_alloc = 0.0

    for i, date in enumerate(common_idx):
        alloc = float(alloc_by_date.get(date, 0.0))
        alloc = max(-config.leverage_cap, min(config.leverage_cap, alloc))

        spy_ret = float(spy_returns.get(date, 0.0))
        ref_ret = float(refiner_returns.get(date, 0.0)) if date in refiner_returns.index else 0.0

        # Portfolio return: (1 - |alloc|) in SPY + alloc in refiners
        spy_weight = 1.0 - abs(alloc)
        port_ret = spy_weight * spy_ret + alloc * ref_ret

        # Transaction costs: 2x per-leg on allocation changes
        alloc_change = abs(alloc - prev_alloc)
        txn_cost = alloc_change * 2 * (config.txn_cost_bps_per_leg / 10_000)

        # Borrow cost on short allocation
        if alloc < 0:
            borrow = abs(alloc) * (config.borrow_cost_bps_per_year / 10_000) / TRADING_DAYS_PER_YEAR
        else:
            borrow = 0.0

        daily_nav_ret = port_ret - txn_cost - borrow
        nav.append(nav[-1] * (1 + daily_nav_ret))
        prev_alloc = alloc

    nav_series = pd.Series(nav[1:], index=common_idx)

    # Compute metrics from daily returns
    daily_rets = nav_series.pct_change().dropna()
    mu = daily_rets.mean()
    sigma = daily_rets.std()
    ann_ret = float(mu * TRADING_DAYS_PER_YEAR)
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0

    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = float(drawdown.min())

    metrics = {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "final_nav": float(nav_series.iloc[-1]) if len(nav_series) > 0 else 1.0,
    }

    return {"nav_series": nav_series, "metrics": metrics}


def compare_to_spy_only(spy_returns: pd.Series) -> dict:
    """Pure SPY buy-and-hold baseline metrics."""
    nav = (1 + spy_returns).cumprod()
    daily_rets = spy_returns

    mu = daily_rets.mean()
    sigma = daily_rets.std()
    ann_ret = float(mu * TRADING_DAYS_PER_YEAR)
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "final_nav": float(nav.iloc[-1]) if len(nav) > 0 else 1.0,
    }
