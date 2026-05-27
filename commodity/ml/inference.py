"""
MOIRAI Global Market Sweep & Discovery.

Logic:
1. Preprocessing: Pulls all liquid symbols.
2. Global Sweep: Loops through every symbol in batches.
3. Model Ranking: Calculates a Global Influence Score (Feature Importance) for everyone.
4. Pairs Analysis: Extracts bidirectional influence for the final cluster.

Hardware: NVIDIA RTX 4060 (8GB VRAM)
"""

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data_preprocessing import run_preprocessing, finalize_multivariate_matrix

# ======================================================================
# Attention Capture Hook
# ======================================================================
attention_store = {}

def _capture_attention_hook(module, args, kwargs, output):
    raw_q, raw_k = args[0], args[1]
    query_var_id = kwargs.get("query_var_id")
    kv_var_id = kwargs.get("kv_var_id")

    with torch.no_grad():
        q = module.q_proj(raw_q)
        k = module.k_proj(raw_k)
        q = module.q_norm(rearrange(q, "... q_len (group hpg dim) -> ... group hpg q_len dim", group=module.num_groups, hpg=module.heads_per_group))
        k = module.k_norm(repeat(k, "... kv_len (group dim) -> ... group hpg kv_len dim", group=module.num_groups, hpg=module.heads_per_group))
        scores = (q @ k.transpose(-2, -1)) * module.softmax_scale
        weights = torch.softmax(scores, dim=-1)
        avg_weights = weights.mean(dim=(-4, -3))

    attention_store["weights"] = avg_weights.detach().cpu()
    attention_store["query_var_id"] = query_var_id.detach().cpu() if query_var_id is not None else None
    attention_store["kv_var_id"] = kv_var_id.detach().cpu() if kv_var_id is not None else None


def compute_influence_matrix(num_variates, target_columns):
    weights = attention_store["weights"][0]
    q_var = attention_store["query_var_id"][0]
    kv_var = attention_store["kv_var_id"][0]
    influence = np.zeros((num_variates, num_variates))
    for qi in range(num_variates):
        q_mask = (q_var == qi)
        if not q_mask.any(): continue
        for kj in range(num_variates):
            kv_mask = (kv_var == kj)
            if not kv_mask.any(): continue
            sub_weights = weights[q_mask][:, kv_mask]
            influence[qi, kj] = sub_weights.mean().item()
    return influence


# ======================================================================
# Main Pipeline
# ======================================================================
def run_global_sweep_pipeline():
    # Configuration
    PREDICTION_LENGTH = 10
    CONTEXT_LENGTH = 200
    PATCH_SIZE = "auto"
    BATCH_SIZE = 20
    MODEL_ID = "Salesforce/moirai-1.1-R-small"

    # 1. Preprocessing
    print("\n[1/6] Running Preprocessing...")
    prep_data = run_preprocessing()
    pivot_df = prep_data["pivot_df"]
    all_symbols = list(pivot_df.columns)
    total_symbols = len(all_symbols)
    print(f"      Total liquid symbols to scan: {total_symbols}")

    # 2. Stage 1: Global Market Sweep
    print(f"\n[2/6] STAGE 1: Global Market Sweep (Feature Importance Ranking)...")
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = MoiraiModule.from_pretrained(MODEL_ID)
    global_influence_map = {}

    for i in range(0, total_symbols, BATCH_SIZE):
        chunk_symbols = all_symbols[i : i + BATCH_SIZE]
        num_in_chunk = len(chunk_symbols)
        print(f"      Batch {i//BATCH_SIZE + 1}/{-(-total_symbols//BATCH_SIZE)} ({num_in_chunk} symbols)...", end="\r")
        model_scan = MoiraiForecast(module=module, prediction_length=PREDICTION_LENGTH, context_length=CONTEXT_LENGTH, patch_size=PATCH_SIZE, num_samples=1, target_dim=num_in_chunk, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        last_attn = model_scan.module.encoder.layers[-1].self_attn
        hook = last_attn.register_forward_hook(_capture_attention_hook, with_kwargs=True)
        chunk_df = pivot_df[chunk_symbols].copy()
        chunk_df.index = pd.DatetimeIndex(chunk_df.index, freq="D")
        scan_ds = PandasDataset(chunk_df, target=chunk_symbols, freq="D")
        predictor_scan = model_scan.create_predictor(batch_size=1, device=device)
        list(predictor_scan.predict(scan_ds))
        hook.remove()
        weights, kv_var = attention_store["weights"][0], attention_store["kv_var_id"][0]
        for j in range(num_in_chunk):
            kv_mask = (kv_var == j)
            if kv_mask.any(): global_influence_map[chunk_symbols[j]] = weights[:, kv_mask].sum().item()
        del model_scan, predictor_scan; gc.collect(); torch.cuda.empty_cache()

    print("\n      Sweep Complete.")
    sorted_influence = sorted(global_influence_map.items(), key=lambda x: x[1], reverse=True)
    discovered_target = sorted_influence[0][0]

    print(f"\n{'=' * 65}")
    print(f"  🏆 GLOBAL FEATURE IMPORTANCE (MARKET LEADERS)")
    print(f"{'=' * 65}")
    for rank, (sym, score) in enumerate(sorted_influence[:30], 1):
        bar = "█" * int(score / sorted_influence[0][1] * 20)
        print(f"  {rank:>2}. {sym:<12} {score:>8.4f} {bar}")
    print(f"{'=' * 65}")

    # 3. Build Final Cluster
    print(f"\n[3/6] Building Final Cluster around Winner: {discovered_target}...")
    final_data = finalize_multivariate_matrix(pivot_df, discovered_target)
    mv_df, target_columns = final_data["multivariate_df"], final_data["target_columns"]
    num_variates = len(target_columns)

    # 4. Stage 2: Final Forecast + Pair Discovery
    print(f"\n[4/6] STAGE 2: Full Probabilistic Forecast + Bidirectional Pairs...")
    model_final = MoiraiForecast(module=module, prediction_length=PREDICTION_LENGTH, context_length=CONTEXT_LENGTH, patch_size=PATCH_SIZE, num_samples=100, target_dim=num_variates, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    last_attn_final = model_final.module.encoder.layers[-1].self_attn
    hook_final = last_attn_final.register_forward_hook(_capture_attention_hook, with_kwargs=True)
    final_ds = PandasDataset(mv_df, target=target_columns, freq="D")
    predictor_final = model_final.create_predictor(batch_size=1, device=device)
    forecasts = list(predictor_final.predict(final_ds))
    hook_final.remove()

    # Influence Matrix for the final cluster
    influence = compute_influence_matrix(num_variates, target_columns)

    # 5. Output
    print(f"\n{'=' * 65}")
    print(f"  🔄 MODEL-DISCOVERED PAIRS FOR {discovered_target}")
    print(f"{'=' * 65}")
    print(f"  {'PAIR':<25} {'A→B':>8} {'B→A':>8} {'RELATION'}")
    print(f"  {'─' * 65}")
    for idx in range(1, min(10, num_variates)):
        sym = target_columns[idx]
        a_to_b, b_to_a = influence[0, idx], influence[idx, 0]
        ratio = a_to_b / (b_to_a + 1e-8)
        rel = "← DRIVER" if ratio > 1.1 else "→ FOLLOWER" if ratio < 0.9 else "↔ MIRROR"
        print(f"  {discovered_target} ↔ {sym:<10} {a_to_b:>8.4f} {b_to_a:>8.4f} {rel}")
    print(f"{'=' * 65}")

    forecast = forecasts[0]
    median_traj = np.median(forecast.samples[:, :, 0], axis=0)
    last_actual = mv_df[discovered_target].iloc[-1]
    k = 5.0
    prob_up = 1.0 / (1.0 + np.exp(-k * (median_traj[0] - last_actual)))

    print(f"\n{'=' * 65}")
    print(f"  🎯 FINAL FORECAST: {discovered_target}")
    print(f"{'=' * 65}")
    print(f"  Last Known Z: {last_actual:+.4f} | P(UP): {prob_up*100:.2f}%")
    print(f"  Day 1 Forecast: {median_traj[0]:+.4f} (Δ={median_traj[0]-last_actual:+.4f})")
    print(f"{'=' * 65}\n")

    return discovered_target


if __name__ == "__main__":
    run_global_sweep_pipeline()
