"""Verify the full pipeline produces nonzero outputs with synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from refiner_strategy.config import AB_WEIGHTS, NEUTRAL_PRED, TICKERS
from refiner_strategy.sizing.schemes import SIZERS, DEFAULT_SCHEMES
from refiner_strategy.evaluation.ab_runner import _init_state, _step_one_day, _summarize


def main() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    basket = TICKERS
    weights = AB_WEIGHTS
    notional = 100.0
    schemes = DEFAULT_SCHEMES

    # Mildly bullish Chronos prediction
    bullish_pred = {
        "q10": -0.008, "q20": -0.004, "q30": -0.001, "q40": 0.001,
        "q50": 0.003, "q60": 0.005, "q70": 0.008, "q80": 0.012,
        "q90": 0.015, "p_up": 0.65,
    }

    state = _init_state(schemes, basket)

    for i in range(20):
        T = dates[i]
        preds = {t: dict(bullish_pred) for t in basket}
        det_today = 1.0 if i % 3 != 0 else 0.0
        actual_rets = {t: 0.002 * (1 if i % 2 == 0 else -1) for t in basket}
        rv_today = {t: 0.015 for t in basket}

        _step_one_day(state, schemes, basket, T, preds, det_today,
                      actual_rets, rv_today, notional, weights)

    results = _summarize(state, notional)

    print("=== 20-day synthetic backtest ===\n")
    any_nonzero = False
    for scheme in DEFAULT_SCHEMES:
        m = results[scheme]
        cum = m["cum_pnl"]
        ann = m["ann_ret"]
        sh = m["sharpe"]
        n = m["n"]
        nz = "OK" if abs(cum) > 1e-10 else "** ZERO **"
        if abs(cum) > 1e-10:
            any_nonzero = True
        print(f"  {scheme:12s}  cum_pnl={cum:+10.6f}  ann_ret={ann:+.4%}  sharpe={sh:+.2f}  n={n}  {nz}")

    print("\n=== DET trade log (VLO, first 5 days) ===\n")
    det_trades = results["DET"]["trades"]
    vlo = det_trades[det_trades["ticker"] == "VLO"].head(5)
    for _, row in vlo.iterrows():
        print(f"  {str(row['date'].date()):>12s}  target={row['target_size']:+8.2f}  "
              f"effective={row['effective_size']:+8.2f}  ret={row['actual_ret']:+.5f}  "
              f"pnl={row['asset_pnl']:+.6f}")

    print("\n=== ENS_VETO trade log (VLO, first 5 days) ===\n")
    ev_trades = results["ENS_VETO"]["trades"]
    vlo_ev = ev_trades[ev_trades["ticker"] == "VLO"].head(5)
    for _, row in vlo_ev.iterrows():
        print(f"  {str(row['date'].date()):>12s}  target={row['target_size']:+8.2f}  "
              f"effective={row['effective_size']:+8.2f}  ret={row['actual_ret']:+.5f}  "
              f"pnl={row['asset_pnl']:+.6f}")

    if not any_nonzero:
        print("\n** PROBLEM: All schemes produced zero PnL! **")
        sys.exit(1)
    else:
        print("\nPipeline verification passed: nonzero PnL produced.")


if __name__ == "__main__":
    main()
