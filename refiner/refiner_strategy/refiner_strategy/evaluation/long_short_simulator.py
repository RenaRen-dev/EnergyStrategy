"""Long/short overlay simulator (no SPY default).

Unlike the SPY-default overlay (``spy_default_simulator``) — which parks unused
capital in SPY and only ever goes long the refiners — this simulator runs a
true **long/short** book: a +1 signal buys the stock, a −1 signal **shorts** it,
and a flat scheme decision simply earns nothing that day.

Returns traded are the **beta-hedged** per-ticker returns (README step 6:
short β×SPY against each leg).  These are already baked into ``trades_df`` as
``actual_ret`` by ``ab_runner`` (which reads ``{ticker}_Hedged_Return``), so the
short leg is implicitly market-neutral.

Costs mirror the SPY-default model: a transaction cost on every per-ticker
position change (one leg), plus an annualised borrow cost on short exposure
held during the day.  PnL uses yesterday's position (H4 invariant).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from refiner_strategy.config import (
    BORROW_COST_BPS_PER_YEAR,
    README_NOTIONAL,
    TRADING_DAYS_PER_YEAR,
)
from refiner_strategy.evaluation.benchmarks import pnl_breakdown


@dataclass
class LongShortConfig:
    """Cost assumptions for the long/short simulation."""
    txn_cost_bps_per_leg: float = 5.0
    borrow_cost_bps_per_year: float = BORROW_COST_BPS_PER_YEAR
    notional: float = README_NOTIONAL  # $ notional for README-style $ PnL reporting


def simulate_long_short(
    trades_df: pd.DataFrame,
    config: LongShortConfig | None = None,
    strategy_notional: float = 100.0,
) -> dict:
    """Simulate a pure long/short refiner book from an ab_runner trade log.

    Parameters
    ----------
    trades_df : trade log from ab_runner with columns
                (date, ticker, target_size, effective_size, actual_ret, ...)
                where ``actual_ret`` is the beta-hedged daily return.
    config : cost assumptions.
    strategy_notional : reference notional used by the upstream sizing schemes.

    Returns
    -------
    dict with 'nav_series' (pd.Series) and 'metrics' (dict).
    """
    if config is None:
        config = LongShortConfig()

    if trades_df.empty:
        return {"nav_series": pd.Series(dtype=float), "metrics": {}}

    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    tickers = sorted(trades_df["ticker"].unique())

    # (date × ticker) matrices of target sizes and beta-hedged returns.
    positions_by_date = trades_df.pivot_table(
        index="date", columns="ticker", values="target_size",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=tickers, fill_value=0.0)
    rets_by_date = trades_df.pivot_table(
        index="date", columns="ticker", values="actual_ret",
        aggfunc="mean", fill_value=0.0,
    ).reindex(columns=tickers, fill_value=0.0)

    dates = positions_by_date.index
    nav = [1.0]
    net_rets: list[float] = []  # daily net return per date (exact, incl. day 1)
    prev_pos_frac: Dict[str, float] = {t: 0.0 for t in tickers}

    for date in dates:
        target_pos_frac = {
            t: float(positions_by_date.loc[date, t]) / strategy_notional
            for t in tickers
        }

        # ── Returns earned today on YESTERDAY's positions (H4 invariant) ──
        gross_port_ret = 0.0
        for t in tickers:
            r = rets_by_date.loc[date, t]
            if not pd.isna(r):
                gross_port_ret += prev_pos_frac[t] * float(r)

        # ── Transaction cost on the close-of-day rebalance (one leg/ticker) ──
        ticker_legs = sum(abs(target_pos_frac[t] - prev_pos_frac[t]) for t in tickers)
        txn_cost = ticker_legs * (config.txn_cost_bps_per_leg / 10_000)

        # ── Borrow cost on yesterday's short exposure (held during the day) ──
        prev_short_exposure = sum(abs(p) for p in prev_pos_frac.values() if p < 0)
        borrow = (
            prev_short_exposure
            * (config.borrow_cost_bps_per_year / 10_000)
            / TRADING_DAYS_PER_YEAR
        )

        daily_nav_ret = gross_port_ret - txn_cost - borrow
        net_rets.append(daily_nav_ret)
        nav.append(nav[-1] * (1 + daily_nav_ret))

        prev_pos_frac = target_pos_frac

    nav_series = pd.Series(nav[1:], index=dates)

    # Exact daily net returns (includes day 1, unlike nav.pct_change()).
    daily_rets = pd.Series(net_rets, index=dates)
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
        "total_return": float(nav_series.iloc[-1] - 1.0) if len(nav_series) > 0 else 0.0,
        "n": int(len(nav_series)),
    }
    # README reporting: total/per-year $ PnL on $10M, positive years, invested-Sharpe.
    metrics.update(pnl_breakdown(daily_rets, notional=config.notional))

    return {"nav_series": nav_series, "metrics": metrics}
