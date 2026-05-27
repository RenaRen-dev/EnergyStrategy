"""Smoke test: Chronos-2 fit() with LoRA on tiny data."""
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
        "amazon/chronos-2", device_map="cpu", dtype=torch.float32,
    )

    # Build synthetic training data: 3 series, each (2 variates, 200 timesteps)
    np.random.seed(42)
    train_contexts = []
    for i in range(3):
        target = np.cumsum(np.random.randn(200) * 0.01)
        cov = np.random.randn(200) * 0.5
        ctx = np.stack([target, cov])  # (2, 200)
        train_contexts.append(ctx)

    print(f"Training on {len(train_contexts)} series, each shape {train_contexts[0].shape}")

    # Test 1: try list of 2D arrays (current code)
    print("\nTest 1: fit() with list of 2D arrays...")
    try:
        new_pipeline = pipeline.fit(
            train_contexts,
            prediction_length=1,
            finetune_mode="lora",
            learning_rate=1e-5,
            num_steps=3,  # tiny for smoke test
            batch_size=2,
        )
        print("  SUCCESS: fit() accepted list of 2D arrays")
        print(f"  Returned type: {type(new_pipeline).__name__}")
    except Exception as e:
        print(f"  FAILED: {e}")

        # Test 2: try list of 3D arrays (with batch dim)
        print("\nTest 2: fit() with list of 3D arrays...")
        train_3d = [ctx[np.newaxis, ...] for ctx in train_contexts]
        try:
            new_pipeline = pipeline.fit(
                train_3d,
                prediction_length=1,
                finetune_mode="lora",
                learning_rate=1e-5,
                num_steps=3,
                batch_size=2,
            )
            print("  SUCCESS: fit() accepted list of 3D arrays")
        except Exception as e2:
            print(f"  FAILED: {e2}")

            # Test 3: try single 3D array stacked
            print("\nTest 3: fit() with single 3D array...")
            stacked = np.stack(train_contexts)  # (3, 2, 200)
            try:
                new_pipeline = pipeline.fit(
                    stacked,
                    prediction_length=1,
                    finetune_mode="lora",
                    learning_rate=1e-5,
                    num_steps=3,
                    batch_size=2,
                )
                print("  SUCCESS: fit() accepted 3D stacked array")
            except Exception as e3:
                print(f"  FAILED: {e3}")
                sys.exit(1)

    # Verify the finetuned pipeline can predict
    print("\nVerifying finetuned pipeline predicts...")
    test_ctx = train_contexts[0][np.newaxis, ...]  # (1, 2, 200)
    q_list, _ = new_pipeline.predict_quantiles(
        test_ctx, prediction_length=1,
    )
    q_vals = q_list[0][0, 0, :].numpy()
    print(f"  Quantiles: {q_vals}")
    print(f"  Nonzero: {np.any(q_vals != 0)}")

    print("\nFit smoke test PASSED.")


if __name__ == "__main__":
    main()
