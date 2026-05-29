"""17-fold purged walk-forward LoRA fine-tuning loop for Chronos-2.

Each fold:
  1. Carves a 24-month training window and 6-month test window
  2. Inserts a 5-day purge gap to prevent information leakage
  3. Fine-tunes Chronos-2 with LoRA adapters
  4. Generates quantile predictions on the test set

Key correctness invariants:
  H3 — fold_seed = LORA_BASE_SEED + fold_idx (deterministic per-fold)
  Strict < — history never includes the prediction day
  Purge gap — train_end = test_start - WFO_PURGE_DAYS - 1

Chronos-2 API notes (chronos-forecasting >= 1.4):
  - Use Chronos2Pipeline, NOT BaseChronosPipeline (only Chronos2Pipeline has fit)
  - fit() returns a NEW pipeline; does not mutate in-place
  - fit() accepts finetune_mode="lora", not optim= or seed= kwargs
  - predict_quantiles() returns (list[Tensor], list[Tensor])
    where each tensor shape is (n_variates, prediction_length, n_quantiles)
  - Variate 0 is the target; variate 1+ are covariates
"""
from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Patch torch.optim.AdamW to disable fused on Mac/CPU (not CUDA-compatible).
# Must run before any trainer/Chronos code tries to create an optimizer with fused=True.
import torch
_original_adamw_init = torch.optim.AdamW.__init__

def _patched_adamw_init(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                        amsgrad=False, fused=None, **kwargs):
    if fused is not None and torch.cuda.is_available() is False:
        # On CPU/Mac, force fused=False; on CUDA, let the caller decide
        fused = False
    _original_adamw_init(self, params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         amsgrad=amsgrad, fused=fused, **kwargs)

torch.optim.AdamW.__init__ = _patched_adamw_init

from refiner_strategy.config import (
    CHRONOS_MODEL_ID,
    CONTEXT_LENGTH,
    LORA_BASE_SEED,
    NEUTRAL_PRED,
    OOS_TEST_END,
    OOS_TEST_START,
    TICKERS,
    WFO_PURGE_DAYS,
    WFO_TEST_MONTHS,
    WFO_TRAIN_MONTHS,
)
from refiner_strategy.utils.torch_helpers import select_device


def _fold_boundaries(
    master_start: str,
    master_end: str,
) -> List[dict]:
    """Generate non-overlapping 6-month test windows starting at OOS_TEST_START.

    Each fold:
      test_start, test_end = 6-month window
      train_end = test_start - WFO_PURGE_DAYS - 1  (purge gap)
      train_start = train_end - 24 months
    """
    folds = []
    current = pd.Timestamp(OOS_TEST_START)
    end = pd.Timestamp(master_end)
    fold_idx = 0

    while current < end:
        test_start = current
        test_end = test_start + pd.DateOffset(months=WFO_TEST_MONTHS) - timedelta(days=1)
        if test_end > end:
            test_end = end

        train_end = test_start - timedelta(days=WFO_PURGE_DAYS + 1)
        train_start = train_end - pd.DateOffset(months=WFO_TRAIN_MONTHS)

        folds.append(
            {
                "fold_idx": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        current = test_end + timedelta(days=1)
        fold_idx += 1

    return folds


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility (H3 fix)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_one_fold(
    master: pd.DataFrame,
    fold: dict,
    output_dir: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """Fine-tune Chronos-2 on one fold and generate test predictions.

    H3 FIX: fold_seed = LORA_BASE_SEED + fold_idx, set at start.
    """
    fold_idx = fold["fold_idx"]
    out_path = output_dir / f"fold_{fold_idx:02d}.parquet"

    # Caching: skip if already computed
    if out_path.exists():
        return pd.read_parquet(out_path)

    fold_seed = LORA_BASE_SEED + fold_idx
    _set_seed(fold_seed)

    import torch
    from chronos import Chronos2Pipeline

    # Must use Chronos2Pipeline (not BaseChronosPipeline) for fit()
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    )

    # --- Training phase ---
    train_data = master.loc[fold["train_start"]:fold["train_end"]]

    # Build training contexts: list of 2D arrays (n_variates, n_timesteps)
    # Variate 0 = target (hedged return * 100), Variate 1 = covariate (crack Z)
    train_contexts = []
    for ticker in TICKERS:
        hedged_col = f"{ticker}_Hedged_Return"
        if hedged_col not in train_data.columns:
            continue
        target = train_data[hedged_col].dropna().values
        cov = train_data["Crack_Z_Score"].dropna().values
        min_len = min(len(target), len(cov))
        if min_len == 0:
            continue
        ctx = np.stack([target[-min_len:] * 100, cov[-min_len:]])
        train_contexts.append(ctx)

    if train_contexts:
        # fit() returns a NEW pipeline with LoRA adapters applied
        pipeline = pipeline.fit(
            train_contexts,
            prediction_length=1,
            finetune_mode="lora",
            learning_rate=1e-5,
            num_steps=200,
            batch_size=64,
            output_dir=str(output_dir / f"finetune_fold_{fold_idx:02d}"),
        )

    # --- Test phase ---
    test_dates = master.loc[fold["test_start"]:fold["test_end"]].index
    records = []
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for T in test_dates:
        history = master[master.index < T]  # strict <
        if len(history) < 60:
            continue

        for ticker in TICKERS:
            try:
                hedged_col = f"{ticker}_Hedged_Return"
                hedged_hist = history[hedged_col].dropna().values
                crack_z_hist = history["Crack_Z_Score"].dropna().values

                min_len = min(len(hedged_hist), len(crack_z_hist))
                if min_len == 0:
                    rec = {"Date": T, "Ticker": ticker, **NEUTRAL_PRED}
                    records.append(rec)
                    continue

                hedged_hist = hedged_hist[-min_len:]
                crack_z_hist = crack_z_hist[-min_len:]

                if min_len > CONTEXT_LENGTH:
                    hedged_hist = hedged_hist[-CONTEXT_LENGTH:]
                    crack_z_hist = crack_z_hist[-CONTEXT_LENGTH:]

                # (1, n_variates, n_timesteps) — batch dim required
                ctx = np.stack([hedged_hist * 100, crack_z_hist])
                ctx = ctx[np.newaxis, ...]  # (1, 2, T)

                # predict_quantiles returns (list[Tensor], list[Tensor])
                # Each tensor: (n_variates, prediction_length, n_quantiles)
                q_list, _ = pipeline.predict_quantiles(
                    ctx,
                    prediction_length=1,
                    quantile_levels=quantile_levels,
                )
                # q_list[0][0, 0, :] = variate 0 (target), step 0, all quantiles
                q_vals = q_list[0][0, 0, :].numpy() / 100.0

                if np.any(np.isnan(q_vals)):
                    records.append({"Date": T, "Ticker": ticker, **NEUTRAL_PRED})
                    continue

                rec = {"Date": T, "Ticker": ticker}
                for i, level in enumerate(quantile_levels):
                    rec[f"q{int(level * 100)}"] = float(q_vals[i])
                rec["p_up"] = float(np.mean(q_vals > 0))
                records.append(rec)

            except Exception as e:
                print(f"  [WARN] Fold {fold_idx}, {T.date()}, {ticker}: {e}")
                records.append({"Date": T, "Ticker": ticker, **NEUTRAL_PRED})

    result = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result


def run_all_folds(
    master: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run all walk-forward folds and return concatenated predictions."""
    device = select_device()
    master_start = str(master.index.min().date())
    master_end = str(master.index.max().date())
    folds = _fold_boundaries(master_start, master_end)

    all_preds = []
    for fold in folds:
        pred = run_one_fold(master, fold, output_dir, device=device)
        all_preds.append(pred)

    return pd.concat(all_preds, ignore_index=True)


def load_all_predictions(pred_dir: Path) -> pd.DataFrame:
    """Load all fold_*.parquet files from a prediction directory."""
    files = sorted(pred_dir.glob("fold_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No fold_*.parquet files in {pred_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
