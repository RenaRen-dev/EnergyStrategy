"""Smoke test: load Chronos-2 and verify predict_quantiles works."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from chronos import Chronos2Pipeline


def main() -> None:
    print("Loading Chronos-2...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"  Model loaded: {type(pipeline).__name__}")

    # Synthetic 2-variate context: 100 timesteps
    np.random.seed(42)
    target = np.cumsum(np.random.randn(100) * 0.01)  # random walk
    covariate = np.random.randn(100) * 0.5  # noise covariate
    context = np.stack([target, covariate])  # (2, 100)
    context = context[np.newaxis, ...]  # (1, 2, 100) = (n_series, n_variates, T)

    print(f"  Context shape: {context.shape}")

    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    q_list, m_list = pipeline.predict_quantiles(
        context,
        prediction_length=1,
        quantile_levels=quantile_levels,
    )

    print(f"  q_list type: {type(q_list)}, length: {len(q_list)}")
    print(f"  q_list[0] shape: {q_list[0].shape}")
    print(f"  m_list[0] shape: {m_list[0].shape}")

    # Extract variate 0 (target), step 0, all quantiles
    q_vals = q_list[0][0, 0, :].numpy()
    mean_val = m_list[0][0, 0].item()

    print(f"\n  Quantile values: {q_vals}")
    print(f"  Mean forecast: {mean_val:.6f}")
    print(f"  All finite: {np.all(np.isfinite(q_vals))}")
    print(f"  Any nonzero: {np.any(q_vals != 0)}")

    if np.any(q_vals != 0) and np.all(np.isfinite(q_vals)):
        print("\nChronos-2 smoke test PASSED: nonzero finite predictions.")
    else:
        print("\nChronos-2 smoke test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
