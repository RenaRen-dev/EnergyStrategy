"""Master dataset builder — fetches all data from yfinance at runtime.

Produces a single DataFrame that every downstream module consumes:
crack spread, Z-score, equity returns, betas, and hedged returns.

Key correctness invariants preserved here:
  H1 — unified Z-score across the full crack series (no per-slice resets)
  H5 — rolling beta is lagged by one day (.shift(1)) before use
  yfinance column order — alphabetical, so CL/HO/RB not CL/RB/HO
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from refiner_strategy.config import (
    AB_WEIGHTS,
    BETA_WINDOW,
    DATA_START,
    TICKERS,
    Z_SCORE_WINDOW,
)


def _download_prices(
    symbols: List[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Download adjusted-close prices with basic sanity checks."""
    raw = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbols}")
    if len(symbols) == 1:
        close = raw["Close"].to_frame(symbols[0])
    else:
        close = raw["Close"].copy()
    if close.isna().all().any():
        bad = list(close.columns[close.isna().all()])
        raise RuntimeError(f"All-NaN price columns: {bad}")
    return close


def build_master_dataset(
    start: str = DATA_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Build the master analytics DataFrame from yfinance data.

    Returns
    -------
    DataFrame indexed by date with columns:
        Crack_Spread, Crack_Z_Score, Basket_Return, SPY_Return,
        Rolling_Beta, Beta_Hedged_Return,
        VLO_Hedged_Return, ..., CVI_Hedged_Return
    """
    # ------------------------------------------------------------------
    # 1. Futures prices — crack spread
    # ------------------------------------------------------------------
    futures_tickers = ["CL=F", "RB=F", "HO=F"]
    fut_raw = yf.download(
        futures_tickers, start=start, end=end, progress=False, auto_adjust=False
    )
    if fut_raw.empty:
        raise RuntimeError("yfinance returned empty data for futures tickers")

    # yfinance alphabetical order: [CL=F, HO=F, RB=F]
    fut_close = fut_raw["Close"].copy()
    fut_close.columns = ["CL", "HO", "RB"]
    fut_close = fut_close.dropna()

    crack = (2 * fut_close["RB"] * 42 + fut_close["HO"] * 42) / 3 - fut_close["CL"]

    # H1 FIX: single unified 256-day rolling Z-score on the FULL series
    roll_mean = crack.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW).mean()
    roll_std = crack.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW).std()
    crack_z = ((crack - roll_mean) / roll_std).clip(-3, 3)

    # ------------------------------------------------------------------
    # 2. Equity prices
    # ------------------------------------------------------------------
    equity_symbols = TICKERS + ["SPY"]
    eq_close = _download_prices(equity_symbols, start=start, end=end)
    eq_returns = eq_close.pct_change()

    spy_ret = eq_returns["SPY"]

    # Basket return (cap-weighted)
    basket_ret = sum(
        eq_returns[t] * AB_WEIGHTS[t] for t in TICKERS
    )

    # ------------------------------------------------------------------
    # 3. Rolling beta with H5 lag fix
    # ------------------------------------------------------------------
    def _rolling_beta_lagged(asset_ret: pd.Series, market_ret: pd.Series) -> pd.Series:
        """Compute rolling OLS beta and lag by 1 day (H5 fix)."""
        cov = asset_ret.rolling(BETA_WINDOW).cov(market_ret)
        var = market_ret.rolling(BETA_WINDOW).var()
        beta = cov / var
        return beta.shift(1)  # H5: today's beta must NOT use today's return

    basket_beta_lagged = _rolling_beta_lagged(basket_ret, spy_ret)

    # ------------------------------------------------------------------
    # 4. Hedged returns
    # ------------------------------------------------------------------
    hedged_basket = basket_ret - basket_beta_lagged * spy_ret

    # ------------------------------------------------------------------
    # 5. Assemble output
    # ------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "Crack_Spread": crack,
            "Crack_Z_Score": crack_z,
            "Basket_Return": basket_ret,
            "SPY_Return": spy_ret,
            "Rolling_Beta": basket_beta_lagged,
            "Beta_Hedged_Return": hedged_basket,
        }
    )

    # Per-stock hedged returns
    for t in TICKERS:
        stock_beta_lagged = _rolling_beta_lagged(eq_returns[t], spy_ret)
        out[f"{t}_Hedged_Return"] = eq_returns[t] - stock_beta_lagged * spy_ret

    # Drop warmup NaN rows
    out = out.dropna(subset=["Crack_Z_Score", "Rolling_Beta"])

    return out
