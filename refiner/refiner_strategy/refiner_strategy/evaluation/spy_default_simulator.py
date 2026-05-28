"""SPY-default overlay simulator (Option B — per-ticker accounting).

When the refiner strategy has no position, the capital sits in SPY
(the "default" allocation).  This module answers: "What if we ran
the refiner strategy as a tactical overlay on top of a passive SPY
portfolio?"

The key insight is that unused capital earns the market return instead
of sitting idle, which is a more realistic comparison than standalone
refiner returns vs SPY buy-and-hold.

Accounting model
----------------
Refiner positions are tracked individually per ticker, NOT as a single
cap-weighted basket.  This matters because each sizing scheme produces
seven independent position decisions that frequently deviate from the
cap weights (a long-only basket would have weight VLO=0.25, MPC=0.25,
...; the strategy might actually hold VLO=+0.25, MPC=-0.10, PSX=+0.20).

Transaction costs are charged on:
  1. Every individual ticker position change (one leg each)
  2. The SPY weight change implied by the net allocation shift (one leg)

Portfolio returns are computed from actual held positions and the actual
per-ticker daily returns, not from a fixed cap-weighted basket return.

PnL accounting uses positions held during the day (yesterday's target),
not today's target — consistent with the H4 invariant in ab_runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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


def fetch_per_ticker_returns(test_end: str | None = None) -> pd.DataFrame:
    """Download TICKERS prices and return per-ticker daily returns as a DataFrame.

    Columns: one per ticker.  Rows: dates.  Values: pct_change daily returns.
    """
    raw = yf.download(TICKERS, start="2014-01-01", end=test_end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for equity tickers")
    close = raw["Close"]
    rets = close.pct_change().dropna()
    return rets


def fetch_unhedged_returns(test_end: str | None = None) -> pd.Series:
    """Cap-weighted basket return (kept for backward compatibility / diagnostic use)."""
    rets = fetch_per_ticker_returns(test_end=test_end)
    basket = sum(rets[t] * AB_WEIGHTS[t] for t in TICKERS)
    return basket


def simulate_spy_default(
    spy_returns: pd.Series,
    ticker_returns: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: SpyDefaultConfig | None = None,
    strategy_notional: float = 100.0,
) -> dict:
    """Simulate a SPY-default portfolio with refiner overlay using per-ticker accounting.

    Parameters
    ----------
    spy_returns : daily SPY returns
    ticker_returns : DataFrame of daily per-ticker returns (columns = ticker symbols)
    trades_df : trade log from ab_runner with columns
                (date, ticker, target_size, effective_size, actual_ret, ...)
    config : cost assumptions
    strategy_notional : reference notional used by the upstream sizing schemes

    Returns
    -------
    dict with 'nav_series' (pd.Series) and 'metrics' (dict)
    """
    if config is None:
        config = SpyDefaultConfig()

    # Pivot trades into a (date × ticker) matrix of target sizes
    tickers = sorted(trades_df["ticker"].unique())
    positions_by_date = trades_df.pivot_table(
        index="date",
        columns="ticker",
        values="target_size",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=tickers, fill_value=0.0)

    common_idx = spy_returns.index.intersection(positions_by_date.index)
    if common_idx.empty:
        return {"nav_series": pd.Series(dtype=float), "metrics": {}}

    nav = [1.0]
    prev_pos_frac: Dict[str, float] = {t: 0.0 for t in tickers}
    prev_spy_weight = 1.0  # start fully in SPY

    for date in common_idx:
        # Target positions for this date, expressed as fractions of notional
        target_pos_frac = {
            t: float(positions_by_date.loc[date, t]) / strategy_notional
            for t in tickers
        }

        # Net allocation to refiners, clipped to the leverage cap
        target_alloc = sum(target_pos_frac.values())
        target_alloc_clipped = max(
            -config.leverage_cap, min(config.leverage_cap, target_alloc)
        )
        target_spy_weight = 1.0 - abs(target_alloc_clipped)

        # ── Returns earned today (using YESTERDAY's positions — H4 invariant) ──
        spy_ret = float(spy_returns.get(date, 0.0))
        refiner_contrib = 0.0
        for t in tickers:
            if t in ticker_returns.columns and date in ticker_returns.index:
                t_ret = ticker_returns.loc[date, t]
                if not pd.isna(t_ret):
                    refiner_contrib += prev_pos_frac[t] * float(t_ret)
        spy_contrib = prev_spy_weight * spy_ret
        gross_port_ret = refiner_contrib + spy_contrib

        # ── Transaction costs on the close-of-day rebalance ──
        # Per-ticker: each position change is one leg
        ticker_legs = sum(
            abs(target_pos_frac[t] - prev_pos_frac[t]) for t in tickers
        )
        ticker_cost = ticker_legs * (config.txn_cost_bps_per_leg / 10_000)

        # SPY: the implied weight change is one leg
        spy_cost = abs(target_spy_weight - prev_spy_weight) * (
            config.txn_cost_bps_per_leg / 10_000
        )

        total_txn_cost = ticker_cost + spy_cost

        # ── Borrow cost on yesterday's short positions (held during the day) ──
        prev_short_exposure = sum(abs(p) for p in prev_pos_frac.values() if p < 0)
        borrow = (
            prev_short_exposure
            * (config.borrow_cost_bps_per_year / 10_000)
            / TRADING_DAYS_PER_YEAR
        )

        # ── NAV update ──
        daily_nav_ret = gross_port_ret - total_txn_cost - borrow
        nav.append(nav[-1] * (1 + daily_nav_ret))

        # Roll state forward
        for t in tickers:
            prev_pos_frac[t] = target_pos_frac[t]
        prev_spy_weight = target_spy_weight

    nav_series = pd.Series(nav[1:], index=common_idx)

    # Metrics
    daily_rets = nav_series.pct_change().dropna()
    mu = daily_rets.mean()
    sigma = daily_rets.std()
    ann_ret = float(mu * TRADING_DAYS_PER_YEAR)
    sharpe = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0
    )

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
    sharpe = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0
    )

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "final_nav": float(nav.iloc[-1]) if len(nav) > 0 else 1.0,
    }
