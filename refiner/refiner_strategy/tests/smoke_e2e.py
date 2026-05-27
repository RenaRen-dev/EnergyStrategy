"""End-to-end smoke test: Chronos-2 predictions -> sizing -> nonzero PnL.

Loads the real master dataset, runs zero-shot Chronos-2 on 5 trading days,
and verifies that at least some schemes produce nonzero positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from refiner_strategy.config import (
    AB_WEIGHTS, CHRONOS_MODEL_ID, NEUTRAL_PRED, REALIZED_VOL_WINDOW, TICKERS,
)
from refiner_strategy.evaluation.ab_runner import _init_state, _step_one_day, _summarize
from refiner_strategy.sizing.schemes import DEFAULT_SCHEMES


def main() -> None:
    # Load real master data
    master_path = Path("refiner_strategy/outputs/20260521_221630_myrun/datasets/master.csv")
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Master: {len(master)} rows, {master.index.min().date()} to {master.index.max().date()}")

    # Load Chronos-2
    print("Loading Chronos-2 (zero-shot, CPU)...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID, device_map="cpu", dtype=torch.float32,
    )

    # Pick 5 recent trading days
    test_dates = master.index[-10:-5]
    print(f"Test dates: {[str(d.date()) for d in test_dates]}")

    # Build DET signal (simplified: just use +1 for smoke test)
    det_sig = pd.Series(1.0, index=master.index)
    det_lagged = det_sig.shift(1).fillna(0)

    basket = TICKERS
    weights = AB_WEIGHTS
    notional = 100.0
    schemes = DEFAULT_SCHEMES
    state = _init_state(schemes, basket)
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for T in test_dates:
        history = master[master.index < T]

        # Run Chronos predictions for each ticker
        preds = {}
        for ticker in basket:
            hedged_col = f"{ticker}_Hedged_Return"
            hedged_hist = history[hedged_col].dropna().values[-512:]
            crack_z_hist = history["Crack_Z_Score"].dropna().values[-512:]
            min_len = min(len(hedged_hist), len(crack_z_hist))
            hedged_hist = hedged_hist[-min_len:]
            crack_z_hist = crack_z_hist[-min_len:]

            ctx = np.stack([hedged_hist * 100, crack_z_hist])
            ctx = ctx[np.newaxis, ...]

            try:
                q_list, _ = pipeline.predict_quantiles(
                    ctx, prediction_length=1, quantile_levels=quantile_levels,
                )
                q_vals = q_list[0][0, 0, :].numpy() / 100.0
                pred = {}
                for i, level in enumerate(quantile_levels):
                    pred[f"q{int(level * 100)}"] = float(q_vals[i])
                pred["p_up"] = float(np.mean(q_vals > 0))
                preds[ticker] = pred
            except Exception as e:
                print(f"  WARN: {ticker}: {e}")
                preds[ticker] = dict(NEUTRAL_PRED)

        # Show one prediction
        vlo_pred = preds.get("VLO", {})
        print(f"  {T.date()} VLO: q50={vlo_pred.get('q50', 0):.6f} p_up={vlo_pred.get('p_up', 0):.3f}")

        # Realized vol
        rv_today = {}
        for ticker in basket:
            col = f"{ticker}_Hedged_Return"
            trailing = history[col].iloc[-REALIZED_VOL_WINDOW:]
            rv_today[ticker] = float(trailing.std()) if len(trailing) >= REALIZED_VOL_WINDOW else None

        det_today = float(det_lagged.get(T, 0.0))
        actual_rets = {t: float(master.at[T, f"{t}_Hedged_Return"]) for t in basket}

        _step_one_day(state, schemes, basket, T, preds, det_today, actual_rets, rv_today, notional, weights)

    results = _summarize(state, notional)

    print(f"\n{'Scheme':<12} {'cum_pnl':>10} {'n':>4}")
    print("-" * 30)
    any_nonzero_ai = False
    for scheme in DEFAULT_SCHEMES:
        m = results[scheme]
        nz = "OK" if abs(m["cum_pnl"]) > 1e-10 else "ZERO"
        if scheme not in ("DET",) and abs(m["cum_pnl"]) > 1e-10:
            any_nonzero_ai = True
        print(f"  {scheme:<12} {m['cum_pnl']:>+10.6f}  {m['n']:>4}  {nz}")

    if any_nonzero_ai:
        print("\nE2E smoke test PASSED: AI-dependent schemes produce nonzero PnL.")
    else:
        print("\nE2E smoke test FAILED: AI schemes still zero!")
        sys.exit(1)


if __name__ == "__main__":
    main()
