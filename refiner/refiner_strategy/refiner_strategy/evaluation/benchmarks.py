"""Buy-and-hold benchmarks for the refiner strategy.

Three reference portfolios, all evaluated over the README test window
(``trade_date ≥ TEST_START``):

  - **Single stock** — each of the 7 refiner names held outright.
  - **Combined basket** — the B7 cap-weighted basket (production weights in
    ``AB_WEIGHTS``), i.e. the underlying stocks "combined with percentage".
  - **SPY** — broad-market buy-and-hold.

All return the same metric dict shape as the simulators so the 05/06 scripts
can print strategies and benchmarks side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from refiner_strategy.config import (
    AB_WEIGHTS,
    README_NOTIONAL,
    TEST_START,
    TICKERS,
    TRADING_DAYS_PER_YEAR,
)


def pnl_breakdown(daily_rets: pd.Series, notional: float = README_NOTIONAL) -> dict:
    """README-style P&L reporting from a daily (net) return series.

    Mirrors the README's "Report metrics" list:
      - Total $ PnL on a constant *notional* (additive: Σ dailyret × notional)
      - per-year $ PnL  ("P&L in each year")
      - count of positive years  ("14 of 15 positive years")
      - Sharpe on **invested days only** (``pnl[pnl != 0]``)

    Returns a dict; ``pnl_by_year`` is a Series indexed by calendar year.
    """
    daily_rets = daily_rets.dropna()
    if len(daily_rets) == 0:
        return {
            "total_pnl": 0.0, "pnl_by_year": pd.Series(dtype=float),
            "n_pos_years": 0, "n_years": 0, "sharpe_invested": 0.0,
        }
    idx = pd.to_datetime(daily_rets.index)
    dollar = daily_rets * notional
    by_year = dollar.groupby(idx.year).sum()
    invested = daily_rets[daily_rets != 0]
    sharpe_inv = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * invested.mean() / invested.std())
        if invested.std() > 0 else 0.0
    )
    return {
        "total_pnl": float(dollar.sum()),
        "pnl_by_year": by_year,
        "n_pos_years": int((by_year > 0).sum()),
        "n_years": int(by_year.shape[0]),
        "sharpe_invested": sharpe_inv,
    }


def buy_and_hold_metrics(returns: pd.Series, notional: float = README_NOTIONAL) -> dict:
    """Standard buy-and-hold metrics + README P&L reporting from daily returns."""
    returns = returns.dropna()
    if len(returns) < 2:
        return {"ann_ret": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0, "final_nav": 1.0,
                "total_return": 0.0, "total_pnl": 0.0, "pnl_by_year": pd.Series(dtype=float),
                "n_pos_years": 0, "n_years": 0, "sharpe_invested": 0.0, "n": len(returns)}

    nav = (1 + returns).cumprod()
    mu = returns.mean()
    sigma = returns.std()
    ann_ret = float(mu * TRADING_DAYS_PER_YEAR)
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma) if sigma > 0 else 0.0

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())

    out = {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] - 1.0),
        "n": int(len(returns)),
    }
    out.update(pnl_breakdown(returns, notional=notional))
    return out


def _slice(returns: pd.Series | pd.DataFrame, start: str) -> pd.Series | pd.DataFrame:
    """Restrict to dates ≥ *start* (README test window)."""
    return returns.loc[returns.index >= pd.Timestamp(start)]


def single_stock_benchmarks(
    ticker_returns: pd.DataFrame,
    start: str = TEST_START,
) -> dict[str, dict]:
    """Buy-and-hold metrics for each underlying refiner stock individually."""
    sliced = _slice(ticker_returns, start)
    out: dict[str, dict] = {}
    for t in TICKERS:
        if t in sliced.columns:
            out[t] = buy_and_hold_metrics(sliced[t])
    return out


def basket_benchmark(
    ticker_returns: pd.DataFrame,
    start: str = TEST_START,
) -> dict:
    """Buy-and-hold metrics for the cap-weighted B7 basket (AB_WEIGHTS)."""
    sliced = _slice(ticker_returns, start)
    basket = sum(sliced[t] * AB_WEIGHTS[t] for t in TICKERS if t in sliced.columns)
    return buy_and_hold_metrics(basket)


def spy_benchmark(
    spy_returns: pd.Series,
    start: str = TEST_START,
) -> dict:
    """Buy-and-hold metrics for SPY."""
    return buy_and_hold_metrics(_slice(spy_returns, start))
