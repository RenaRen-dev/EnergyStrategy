"""Master dataset builder — fetches all data from yfinance at runtime.

Produces a single DataFrame that every downstream module consumes:
crack spread, Z-score, equity returns, betas, and hedged returns.

Key correctness invariants preserved here:
  H1 — unified Z-score across the full crack series (no per-slice resets)
  H5 — rolling beta is lagged by one day (.shift(1)) before use
  yfinance column order — alphabetical, so CL/HO/RB not CL/RB/HO
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from refiner_strategy.config import (
    AB_WEIGHTS,
    BETA_WINDOW,
    DATA_START,
    LOCAL_DATA_DIR,
    TENOR_PROMPT,
    TICKERS,
    Z_SCORE_WINDOW,
)
from refiner_strategy.data.futures_loader import (
    load_fixed_tenor_crack_with_contracts,
    load_real_m3_crack,
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


def _load_local_close(symbol: str) -> pd.Series | None:
    """Load Close series for *symbol* from local cache or bundle.

    Checks cache dir first (user downloads via script 00), then falls back to
    the bundled CSV (≤2021). Returns None if neither exists, so the caller can
    fall back to yfinance.
    """
    # Try cache first
    cache_path = Path(LOCAL_DATA_DIR).parent / "cache" / f"{symbol}_daily.csv"
    for path in [cache_path, LOCAL_DATA_DIR / f"{symbol}_daily.csv"]:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            if "Close" in df.columns:
                return df["Close"].astype(float).sort_index()
        except (ValueError, KeyError, pd.errors.ParserError):
            continue
    return None


def _fetch_recent_close(symbol: str, start: str, end: str | None) -> pd.Series:
    """Yahoo Finance Close for one symbol as a Series (robust to column shape)."""
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty or "Close" not in raw.columns:
        return pd.Series(dtype=float, name=symbol)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # single ticker → first (only) column
    return close.astype(float)


def _load_equity_close_local_plus_yf(
    symbols: List[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Close prices using bundled CSVs for history and yfinance for recent data.

    For each symbol: take the local CSV (≤2021) and append yfinance bars for
    every date after the CSV's last day (≥2022) up to *end*.  Symbols without a
    bundled CSV fall back entirely to yfinance.  This keeps the training-era
    history shape from the in-house export while still extending through the
    live test period (README data-range requirement).

    The bundle and yfinance use different adjustment bases (the bundle is
    dividend-adjusted to its snapshot date, yfinance's auto_adjust=False is not).
    To avoid a spurious return at the junction, the local segment is rescaled by
    the price ratio at the last overlapping date so the two segments are
    continuous on yfinance's current basis.
    """
    series: dict[str, pd.Series] = {}
    yf_only: List[str] = []

    for sym in symbols:
        local = _load_local_close(sym)
        if local is None:
            yf_only.append(sym)
            continue
        local = local.loc[local.index >= pd.Timestamp(start)]
        if local.empty:
            # Local CSV doesn't cover the start date; use yfinance only for this symbol.
            yf_only.append(sym)
            continue
        last_local = local.index.max()

        # Fetch with a short lookback so we get an overlap day to rescale on.
        overlap_start = (last_local - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        recent = _fetch_recent_close(sym, start=overlap_start, end=end)

        if not recent.empty:
            common = local.index.intersection(recent.index)
            if len(common) > 0:
                ref = common.max()
                scale = recent.loc[ref] / local.loc[ref]
                if pd.notna(scale) and scale > 0:
                    local = local * scale
            recent = recent.loc[recent.index > last_local]

        combined = pd.concat([local, recent])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        series[sym] = combined

    if yf_only:
        yf_close = _download_prices(yf_only, start=start, end=end)
        for sym in yf_only:
            series[sym] = yf_close[sym]

    close = pd.DataFrame(series)
    close = close.loc[close.index >= pd.Timestamp(start)]
    if end is not None:
        close = close.loc[close.index <= pd.Timestamp(end)]
    return close.sort_index()


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
    # 1. Futures prices — M3 crack spread.
    #    Priority: real M3 (EIA 2015→2024-04 stitched with recent Databento)
    #    → per-contract hybrid (real Mn where available, continuous elsewhere).
    # ------------------------------------------------------------------
    try:
        crack = load_real_m3_crack(start=start, end=end)["Crack_Spread"]
    except (FileNotFoundError, RuntimeError):
        crack = load_fixed_tenor_crack_with_contracts(
            start=start, end=end, tenor_prompt=TENOR_PROMPT
        )["Crack_Spread"]

    # H1 FIX: single unified 256-day rolling Z-score on the FULL series
    roll_mean = crack.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW).mean()
    roll_std = crack.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW).std()
    crack_z = ((crack - roll_mean) / roll_std).clip(-3, 3)

    # ------------------------------------------------------------------
    # 2. Equity prices
    # ------------------------------------------------------------------
    equity_symbols = TICKERS + ["SPY"]
    eq_close = _load_equity_close_local_plus_yf(equity_symbols, start=start, end=end)
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
