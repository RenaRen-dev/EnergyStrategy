"""A/B harness for comparing sizing schemes.

Two execution modes share identical accounting logic:
  - run_ab_zero_shot: live Chronos inference (no fine-tuning)
  - replay_with_predictions: replays precomputed fold predictions

Correctness invariants:
  H4 — trade log records both target_size and effective_size
  Strict < — history never includes the current day
  Transaction costs applied on position CHANGES, not levels
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from refiner_strategy.config import (
    AB_BASKET,
    AB_WEIGHTS,
    CONTEXT_LENGTH,
    NEUTRAL_PRED,
    REALIZED_VOL_WINDOW,
    TICKERS,
    TRANSACTION_COST_BPS,
)
from refiner_strategy.evaluation.metrics import horizon_aligned_hit_rate, metrics_from_pnl
from refiner_strategy.sizing.schemes import DEFAULT_SCHEMES, SIZERS


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _init_state(
    schemes: Sequence[str],
    basket: List[str],
) -> Dict[str, dict]:
    """Initialise per-scheme accounting state."""
    state = {}
    for scheme in schemes:
        state[scheme] = {
            "positions": {t: 0.0 for t in basket},
            "daily_pnl": [],
            "trades": [],
            "txn_cost_total": 0.0,
        }
    return state


def _step_one_day(
    state: Dict[str, dict],
    schemes: Sequence[str],
    basket: List[str],
    date: pd.Timestamp,
    preds: Dict[str, dict],
    det_today: float,
    actual_rets: Dict[str, float],
    rv_today: Dict[str, float | None],
    notional: float,
    weights: Dict[str, float],
) -> None:
    """Advance every scheme by one trading day."""
    for scheme in schemes:
        sizer = SIZERS[scheme]
        day_pnl = 0.0
        for ticker in basket:
            pred = preds.get(ticker, dict(NEUTRAL_PRED))
            rv = rv_today.get(ticker)

            target_size = sizer(pred, det_today, weights, ticker, notional, rv)
            effective_size = state[scheme]["positions"][ticker]

            friction = abs(target_size - effective_size) * (TRANSACTION_COST_BPS / 10_000)
            actual_ret = actual_rets.get(ticker, 0.0)
            asset_pnl = effective_size * actual_ret - friction

            day_pnl += asset_pnl
            state[scheme]["txn_cost_total"] += friction

            state[scheme]["trades"].append(
                {
                    "date": date,
                    "ticker": ticker,
                    "scheme": scheme,
                    "target_size": target_size,
                    "effective_size": effective_size,
                    "actual_ret": actual_ret,
                    "asset_pnl": asset_pnl,
                }
            )

            state[scheme]["positions"][ticker] = target_size

        state[scheme]["daily_pnl"].append({"date": date, "pnl": day_pnl})


def _summarize(state: Dict[str, dict], notional: float) -> Dict[str, dict]:
    """Build metrics dict from accumulated state."""
    results = {}
    for scheme, s in state.items():
        pnl_df = pd.DataFrame(s["daily_pnl"])
        if pnl_df.empty:
            pnl_series = pd.Series(dtype=float)
        else:
            pnl_series = pnl_df.set_index("date")["pnl"]

        trades_df = pd.DataFrame(s["trades"])
        hit_rate = horizon_aligned_hit_rate(trades_df) if not trades_df.empty else float("nan")

        m = metrics_from_pnl(pnl_series, notional=notional)
        m["daily_pnl"] = pnl_series
        m["trades"] = trades_df
        m["hit_rate"] = hit_rate
        m["txn_cost"] = s["txn_cost_total"]
        m["num_trades"] = len(trades_df)
        results[scheme] = m
    return results


# ---------------------------------------------------------------------------
# Live Chronos prediction
# ---------------------------------------------------------------------------

def _live_chronos_predict(
    pipeline: object,
    history: pd.DataFrame,
    ticker: str,
) -> dict:
    """Run Chronos inference for a single ticker on historical data."""
    try:
        import torch

        hedged_col = f"{ticker}_Hedged_Return"
        if hedged_col not in history.columns or "Crack_Z_Score" not in history.columns:
            return dict(NEUTRAL_PRED)

        hedged_hist = history[hedged_col].dropna().values
        crack_z_hist = history["Crack_Z_Score"].dropna().values

        min_len = min(len(hedged_hist), len(crack_z_hist))
        if min_len == 0:
            return dict(NEUTRAL_PRED)

        hedged_hist = hedged_hist[-min_len:]
        crack_z_hist = crack_z_hist[-min_len:]

        # Trim to context length
        if min_len > CONTEXT_LENGTH:
            hedged_hist = hedged_hist[-CONTEXT_LENGTH:]
            crack_z_hist = crack_z_hist[-CONTEXT_LENGTH:]

        # (1, n_variates, n_timesteps) — batch dim required
        context = np.stack([hedged_hist * 100, crack_z_hist])
        context = context[np.newaxis, ...]  # (1, 2, T)

        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # predict_quantiles returns (list[Tensor], list[Tensor])
        # Each tensor: (n_variates, prediction_length, n_quantiles)
        q_list, _ = pipeline.predict_quantiles(
            context,
            prediction_length=1,
            quantile_levels=quantile_levels,
        )

        # q_list[0][0, 0, :] = variate 0 (target), step 0, all quantiles
        q_vals = q_list[0][0, 0, :].numpy() / 100.0

        if np.any(np.isnan(q_vals)):
            return dict(NEUTRAL_PRED)

        result = {
            f"q{int(level * 100)}": float(q_vals[i])
            for i, level in enumerate(quantile_levels)
        }
        result["p_up"] = float(np.mean(q_vals > 0))
        return result

    except Exception:
        return dict(NEUTRAL_PRED)


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

def run_ab_zero_shot(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    det_lagged: pd.Series,
    schemes: Sequence[str] = DEFAULT_SCHEMES,
    basket: List[str] | None = None,
    weights: Dict[str, float] | None = None,
    notional: float = 100.0,
    pipeline: object | None = None,
) -> Dict[str, dict]:
    """Run A/B test with live (zero-shot) Chronos predictions."""
    if basket is None:
        basket = AB_BASKET
    if weights is None:
        weights = AB_WEIGHTS

    state = _init_state(schemes, basket)
    test_dates = df.loc[start_date:end_date].index

    for T in test_dates:
        history = df[df.index < T]  # strict < (no peek)
        if len(history) < 60:
            continue

        # Per-ticker predictions
        preds: Dict[str, dict] = {}
        if pipeline is not None:
            for ticker in basket:
                preds[ticker] = _live_chronos_predict(pipeline, history, ticker)
        else:
            for ticker in basket:
                preds[ticker] = dict(NEUTRAL_PRED)

        # Realised vol from trailing returns
        rv_today: Dict[str, float | None] = {}
        for ticker in basket:
            col = f"{ticker}_Hedged_Return"
            if col in history.columns:
                trailing = history[col].iloc[-REALIZED_VOL_WINDOW:]
                rv_today[ticker] = float(trailing.std()) if len(trailing) >= REALIZED_VOL_WINDOW else None
            else:
                rv_today[ticker] = None

        det_today = float(det_lagged.get(T, 0.0))

        actual_rets: Dict[str, float] = {}
        for ticker in basket:
            col = f"{ticker}_Hedged_Return"
            if col in df.columns and T in df.index:
                actual_rets[ticker] = float(df.at[T, col])
            else:
                actual_rets[ticker] = 0.0

        _step_one_day(
            state, schemes, basket, T, preds, det_today,
            actual_rets, rv_today, notional, weights,
        )

    return _summarize(state, notional)


def replay_with_predictions(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    det_lagged: pd.Series,
    schemes: Sequence[str] = DEFAULT_SCHEMES,
    basket: List[str] | None = None,
    weights: Dict[str, float] | None = None,
    notional: float = 100.0,
) -> Dict[str, dict]:
    """Replay A/B test using precomputed predictions from walk-forward folds."""
    if basket is None:
        basket = AB_BASKET
    if weights is None:
        weights = AB_WEIGHTS

    # Build O(1) lookup: (Date, Ticker) -> row dict
    preds_map: Dict[Tuple[pd.Timestamp, str], dict] = {}
    for _, row in preds.iterrows():
        key = (pd.Timestamp(row["Date"]), row["Ticker"])
        preds_map[key] = row.to_dict()

    state = _init_state(schemes, basket)
    test_dates = sorted(set(preds["Date"]))

    for T in test_dates:
        T = pd.Timestamp(T)
        if T not in df.index:
            continue

        history = df[df.index < T]  # strict <
        if len(history) < 60:
            continue

        # Read predictions from precomputed map
        day_preds: Dict[str, dict] = {}
        for ticker in basket:
            key = (T, ticker)
            if key in preds_map:
                row = preds_map[key]
                try:
                    pred = {}
                    for k in NEUTRAL_PRED:
                        if k in row:
                            v = row[k]
                            fv = float(v)
                            pred[k] = 0.0 if (fv != fv) else fv  # NaN != NaN
                        else:
                            pred[k] = NEUTRAL_PRED[k]
                    day_preds[ticker] = pred
                except (TypeError, ValueError):
                    day_preds[ticker] = dict(NEUTRAL_PRED)
            else:
                day_preds[ticker] = dict(NEUTRAL_PRED)

        # Realised vol
        rv_today: Dict[str, float | None] = {}
        for ticker in basket:
            col = f"{ticker}_Hedged_Return"
            if col in history.columns:
                trailing = history[col].iloc[-REALIZED_VOL_WINDOW:]
                rv_today[ticker] = float(trailing.std()) if len(trailing) >= REALIZED_VOL_WINDOW else None
            else:
                rv_today[ticker] = None

        det_today = float(det_lagged.get(T, 0.0))

        actual_rets: Dict[str, float] = {}
        for ticker in basket:
            col = f"{ticker}_Hedged_Return"
            if col in df.columns:
                actual_rets[ticker] = float(df.at[T, col])
            else:
                actual_rets[ticker] = 0.0

        _step_one_day(
            state, schemes, basket, T, day_preds, det_today,
            actual_rets, rv_today, notional, weights,
        )

    return _summarize(state, notional)
