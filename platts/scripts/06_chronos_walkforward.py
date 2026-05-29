"""Walk-forward LoRA fine-tune of Chronos-2 on RB.

Adapts refiner_strategy/finetune/walkforward.py for single-product futures:
  - Target = RB_LogReturn (one series, no per-ticker loop)
  - Covariates = all *_Z columns in master_dataset.parquet (MOIRAI top-K)
  - Same purged WFO geometry: 24-mo train, 6-mo test, 5-day purge

Per fold:
  1. Carve train [test_start - purge - 24mo, test_start - purge - 1]
  2. Fine-tune Chronos-2 with LoRA on (target, *covariates) context
  3. For each test day, predict q10..q90 + p_up using history < T

Outputs:
    platts/outputs/chronos_predictions/fold_NN.parquet
    platts/outputs/chronos_predictions/all_preds.parquet  (concat after run)

Usage:
    python platts/scripts/06_chronos_walkforward.py             # 200 steps/fold
    python platts/scripts/06_chronos_walkforward.py --fast      # 50 steps, smoke
    python platts/scripts/06_chronos_walkforward.py --device cpu

CPU runtime estimate (Apple Silicon, 200 steps, 7 folds): 1.5-2 hours.
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_PATH = PROJECT_ROOT / "platts" / "outputs" / "master_dataset.parquet"
PRED_DIR    = PROJECT_ROOT / "platts" / "outputs" / "chronos_predictions"

# ---------------------------------------------------------------------------
# Config (matches refiner_strategy/config.py)
# ---------------------------------------------------------------------------
CHRONOS_MODEL_ID = "amazon/chronos-2"
CONTEXT_LENGTH   = 512
WFO_TRAIN_MONTHS = 24
WFO_TEST_MONTHS  = 6
WFO_PURGE_DAYS   = 5
OOS_TEST_START   = "2018-04-01"   # ~3 yrs of pre-OOS for the first train window
LORA_BASE_SEED   = 42

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NEUTRAL_PRED = {f"q{int(q*100)}": 0.0 for q in QUANTILES} | {"p_up": 0.5}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def select_device(prefer: str = "auto") -> str:
    """Same logic as MOIRAI script: prefer CUDA, fall back to CPU on Mac."""
    import torch
    if prefer == "cpu":
        return "cpu"
    if prefer == "mps":
        return "mps"  # let user try at own risk
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Fold boundaries — identical to refiner
# ---------------------------------------------------------------------------
def _fold_boundaries(master_end: pd.Timestamp) -> list[dict]:
    folds = []
    current = pd.Timestamp(OOS_TEST_START)
    fold_idx = 0
    while current < master_end:
        test_start = current
        test_end = test_start + pd.DateOffset(months=WFO_TEST_MONTHS) - timedelta(days=1)
        if test_end > master_end:
            test_end = master_end
        train_end = test_start - timedelta(days=WFO_PURGE_DAYS + 1)
        train_start = train_end - pd.DateOffset(months=WFO_TRAIN_MONTHS)
        folds.append({
            "fold_idx":    fold_idx,
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        current = test_end + timedelta(days=1)
        fold_idx += 1
    return folds


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_one_fold(
    master: pd.DataFrame,
    fold: dict,
    output_dir: Path,
    z_cols: list[str],
    device: str,
    num_steps: int,
    batch_size: int,
) -> pd.DataFrame:
    fold_idx = fold["fold_idx"]
    out_path = output_dir / f"fold_{fold_idx:02d}.parquet"
    if out_path.exists():
        print(f"  [fold {fold_idx}] SKIP (exists)")
        return pd.read_parquet(out_path)

    _set_seed(LORA_BASE_SEED + fold_idx)
    import torch
    from chronos import Chronos2Pipeline

    print(f"  [fold {fold_idx}] loading Chronos-2 on {device} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID, device_map=device, dtype=torch.float32,
    )

    # ----- training context: (target, cov1, cov2, ...) -----
    train = master.loc[fold["train_start"]:fold["train_end"]]
    target = train["RB_LogReturn"].dropna().values * 100.0
    covs   = [train[c].dropna().values for c in z_cols]
    min_len = min(len(target), *(len(c) for c in covs))
    if min_len == 0:
        raise RuntimeError(f"Fold {fold_idx}: no training data")
    target = target[-min_len:]
    covs   = [c[-min_len:] for c in covs]
    train_ctx = np.stack([target, *covs])         # (1+K, min_len)
    train_contexts = [train_ctx]                  # one context for one product

    print(f"  [fold {fold_idx}] LoRA fine-tune: {num_steps} steps, batch={batch_size}, "
          f"train_len={min_len} ...")
    pipeline = pipeline.fit(
        train_contexts,
        prediction_length=1,
        finetune_mode="lora",
        learning_rate=1e-5,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=str(output_dir / f"finetune_fold_{fold_idx:02d}"),
    )

    # ----- test phase: per-day predictions -----
    test_dates = master.loc[fold["test_start"]:fold["test_end"]].index
    records = []
    print(f"  [fold {fold_idx}] inference on {len(test_dates)} test days ...")
    for T in test_dates:
        history = master[master.index < T]              # strict <
        if len(history) < 60:
            records.append({"Date": T, **NEUTRAL_PRED})
            continue

        hedged_hist = history["RB_LogReturn"].dropna().values * 100.0
        cov_hists   = [history[c].dropna().values for c in z_cols]
        min_h = min(len(hedged_hist), *(len(c) for c in cov_hists))
        if min_h == 0:
            records.append({"Date": T, **NEUTRAL_PRED})
            continue
        hedged_hist = hedged_hist[-min_h:]
        cov_hists   = [c[-min_h:] for c in cov_hists]
        if min_h > CONTEXT_LENGTH:
            hedged_hist = hedged_hist[-CONTEXT_LENGTH:]
            cov_hists   = [c[-CONTEXT_LENGTH:] for c in cov_hists]

        ctx = np.stack([hedged_hist, *cov_hists])[np.newaxis, ...]   # (1, 1+K, T)

        try:
            q_list, _ = pipeline.predict_quantiles(
                ctx,
                prediction_length=1,
                quantile_levels=QUANTILES,
            )
            q_vals = q_list[0][0, 0, :].numpy() / 100.0  # undo *100
            if np.any(np.isnan(q_vals)):
                records.append({"Date": T, **NEUTRAL_PRED})
                continue
            rec = {"Date": T}
            for i, level in enumerate(QUANTILES):
                rec[f"q{int(level*100)}"] = float(q_vals[i])
            rec["p_up"] = float(np.mean(q_vals > 0))
            records.append(rec)
        except Exception as e:
            print(f"    [WARN] {T.date()}: {type(e).__name__}: {e}")
            records.append({"Date": T, **NEUTRAL_PRED})

    fold_df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_parquet(out_path, index=False)
    print(f"  [fold {fold_idx}] OK   wrote {out_path.name}  ({len(fold_df)} rows)")
    return fold_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true",
                   help="50 steps/fold, batch=16 — quick smoke test")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override fine-tune steps per fold (default 200)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override fine-tune batch size (default 64)")
    p.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    args = p.parse_args()

    if args.fast:
        num_steps = args.num_steps or 50
        batch_size = args.batch_size or 16
    else:
        num_steps = args.num_steps or 200
        batch_size = args.batch_size or 64

    device = select_device(args.device)
    print(f"[CONFIG] device={device}  num_steps={num_steps}  batch={batch_size}\n")

    print(f"[LOAD] {MASTER_PATH.name} ...")
    master = pd.read_parquet(MASTER_PATH)
    if not isinstance(master.index, pd.DatetimeIndex):
        master.index = pd.to_datetime(master.index)
    master = master.sort_index()

    z_cols = [c for c in master.columns if c.endswith("_Z")]
    print(f"       {len(master)} rows, target=RB_LogReturn, covariates={z_cols}")

    folds = _fold_boundaries(master.index.max())
    print(f"\n[WFO] {len(folds)} folds spanning "
          f"{folds[0]['test_start'].date()} -> {folds[-1]['test_end'].date()}")
    for f in folds:
        print(f"  fold {f['fold_idx']}: "
              f"train [{f['train_start'].date()}..{f['train_end'].date()}]  "
              f"test [{f['test_start'].date()}..{f['test_end'].date()}]")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] Writing predictions to {PRED_DIR}")
    all_dfs = []
    for f in folds:
        try:
            df = run_one_fold(master, f, PRED_DIR, z_cols,
                              device=device, num_steps=num_steps, batch_size=batch_size)
            all_dfs.append(df)
        except Exception as e:
            print(f"  [fold {f['fold_idx']}] FAIL: {type(e).__name__}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True).sort_values("Date").reset_index(drop=True)
        combined_path = PRED_DIR / "all_preds.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\n[OK] Concatenated -> {combined_path}  ({len(combined)} predictions)")
        print("\n  Sample predictions (first 3, last 3):")
        print(combined.head(3).to_string())
        print("...")
        print(combined.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
