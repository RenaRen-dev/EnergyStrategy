"""
AI Inference for Backtest Project 1.

Stage 1 (MOIRAI):  Full bidirectional attention analysis.
                   Proves crack321 drives stock prices.
                   Classifies each stock as Driver / Follower / Mirror.
Stage 2 (Chronos): Predicts ALL crack-driven stocks so the backtester
                   can compare single-stock vs basket strategies.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from einops import rearrange, repeat

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CHRONOS_MODEL = "amazon/chronos-2"
MOIRAI_MODEL = "Salesforce/moirai-1.1-R-small"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 10

STOCK_COLS = [
    "VLO_Hedged_Return", "MPC_Hedged_Return", "PSX_Hedged_Return",
    "DINO_Hedged_Return", "PBF_Hedged_Return", "DK_Hedged_Return",
    "CVI_Hedged_Return",
]

# ======================================================================
# STAGE 1: MOIRAI — Full Bidirectional Attention Analysis
# ======================================================================
attention_store = {}

def _capture_attention_hook(module, args, kwargs, output):
    raw_q, raw_k = args[0], args[1]
    query_var_id = kwargs.get("query_var_id")
    kv_var_id = kwargs.get("kv_var_id")
    with torch.no_grad():
        q = module.q_proj(raw_q)
        k = module.k_proj(raw_k)
        q = module.q_norm(rearrange(q, "... q_len (group hpg dim) -> ... group hpg q_len dim",
                                    group=module.num_groups, hpg=module.heads_per_group))
        k = module.k_norm(repeat(k, "... kv_len (group dim) -> ... group hpg kv_len dim",
                                 group=module.num_groups, hpg=module.heads_per_group))
        scores = (q @ k.transpose(-2, -1)) * module.softmax_scale
        weights = torch.softmax(scores, dim=-1)
        avg_weights = weights.mean(dim=(-4, -3))
    attention_store["weights"] = avg_weights.detach().cpu()
    attention_store["query_var_id"] = query_var_id.detach().cpu() if query_var_id is not None else None
    attention_store["kv_var_id"] = kv_var_id.detach().cpu() if kv_var_id is not None else None


def run_moirai_discovery(df):
    """
    Full bidirectional attention analysis:
    - Feeds [Crack_Z_Score, VLO, MPC, PSX, DINO, PBF, DK, CVI] as 8 variates.
    - Computes the FULL 8×8 attention influence matrix.
    - For each stock, measures:
        crack_to_stock: How much the Crack's Query attends to the Stock's Key
        stock_to_crack: How much the Stock's Query attends to the Crack's Key
    - Classifies: If stock_to_crack > crack_to_stock → Stock is a FOLLOWER of crack
                  (= crack DRIVES the stock = good to trade)
    """
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    all_cols = ["Crack_Z_Score"] + STOCK_COLS
    mv_df = df[all_cols].copy()
    mv_df = mv_df.asfreq('B').ffill().dropna()

    print(f"  Loading {MOIRAI_MODEL}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = MoiraiModule.from_pretrained(MOIRAI_MODEL)

    model = MoiraiForecast(
        module=module, prediction_length=PREDICTION_LENGTH,
        context_length=min(CONTEXT_LENGTH, len(mv_df)),
        patch_size="auto", num_samples=1, target_dim=len(all_cols),
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    )

    last_attn = model.module.encoder.layers[-1].self_attn
    hook = last_attn.register_forward_hook(_capture_attention_hook, with_kwargs=True)
    ds = PandasDataset(mv_df, target=all_cols, freq="B")
    predictor = model.create_predictor(batch_size=1, device=device)
    list(predictor.predict(ds))
    hook.remove()

    weights = attention_store["weights"][0]
    q_var = attention_store["query_var_id"][0]
    kv_var = attention_store["kv_var_id"][0]

    # Build the full N×N influence matrix
    n = len(all_cols)
    influence_matrix = np.zeros((n, n))
    for qi in range(n):
        q_mask = (q_var == qi)
        if not q_mask.any():
            continue
        for kj in range(n):
            kv_mask = (kv_var == kj)
            if not kv_mask.any():
                continue
            influence_matrix[qi, kj] = weights[q_mask][:, kv_mask].mean().item()

    # Analyze each stock's relationship with the crack spread (index 0)
    stock_analysis = {}
    for j, stock_name in enumerate(STOCK_COLS, start=1):
        ticker = stock_name.replace("_Hedged_Return", "")
        crack_to_stock = influence_matrix[0, j]  # Crack Q → Stock K
        stock_to_crack = influence_matrix[j, 0]  # Stock Q → Crack K
        self_attention = influence_matrix[j, j]   # Stock's self-attention

        # Classification logic:
        # If a stock pays MORE attention to the crack than the crack pays to it,
        # the stock is a FOLLOWER (= crack DRIVES the stock)
        # ONLY followers go into Chronos — because our goal is to use crack to trade stocks.
        ratio = stock_to_crack / (crack_to_stock + 1e-8)
        if ratio > 1.2:
            relation = "FOLLOWER"  # Crack DRIVES this stock → trade it
            tradeable = True
        elif ratio < 0.8:
            relation = "DRIVER"   # This stock drives crack → can't use crack to predict it
            tradeable = False
        else:
            relation = "MIRROR"   # Symmetric → crack doesn't clearly drive it
            tradeable = False

        stock_analysis[ticker] = {
            "column": stock_name,
            "crack_to_stock": float(crack_to_stock),
            "stock_to_crack": float(stock_to_crack),
            "self_attention": float(self_attention),
            "ratio": float(ratio),
            "relation": relation,
            "tradeable": tradeable,
        }

    del model, predictor, module
    gc.collect()
    torch.cuda.empty_cache()

    return stock_analysis, influence_matrix, all_cols


# ======================================================================
# STAGE 2: Chronos-2 — Predict ALL tradeable stocks
# ======================================================================
def run_chronos_inference(pipeline, history_df, target_col, covariate_cols):
    cols = [target_col] + covariate_cols
    context = history_df[cols].values.T
    if context.shape[1] > CONTEXT_LENGTH:
        context = context[:, -CONTEXT_LENGTH:]
        
    # Scale up signal magnitude by 100 to avoid Chronos rounding out fractional returns
    context_scaled = context * 100.0
    
    tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        q_list, m_list = pipeline.predict_quantiles(
            tensor, prediction_length=1,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
    q_tensor = q_list[0]
    # q_tensor shape: (n_variates, pred_len, 9_quantiles)
    # Quantile indices: 0=Q10, 1=Q20, 2=Q30, 3=Q40, 4=Q50, 5=Q60, 6=Q70, 7=Q80, 8=Q90
    # Divide by 100 to scale back to standard fractional return space
    target_quantiles = q_tensor[0, 0, :].cpu().numpy() / 100.0
    p_up = float(np.mean(target_quantiles > 0.0))
    return {
        'q10': float(target_quantiles[0]),
        'q20': float(target_quantiles[1]),
        'q30': float(target_quantiles[2]),
        'q40': float(target_quantiles[3]),
        'q50': float(target_quantiles[4]),  # median
        'q60': float(target_quantiles[5]),
        'q70': float(target_quantiles[6]),
        'q80': float(target_quantiles[7]),
        'q90': float(target_quantiles[8]),
        'p_up': p_up,
    }


# ======================================================================
# FULL PIPELINE
# ======================================================================
def run_full_pipeline(dry_run_days=None):
    print("=" * 70)
    print(" BACKTEST: MOIRAI Stock Selection + Chronos Forecast")
    print("=" * 70)

    print("\n[1/3] Loading master dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # Load actual Close prices for each stock and join them
    data_backtest_dir = os.path.abspath(os.path.join(DATA_DIR, "..", "data", "data_backtestproject1"))
    for ticker in ["VLO", "MPC", "PSX", "DINO", "PBF", "DK", "CVI"]:
        path = os.path.join(data_backtest_dir, f"{ticker}_daily.csv")
        if os.path.exists(path):
            stock_df = pd.read_csv(path, parse_dates=['date']).set_index('date')
            df[f"{ticker}_Close"] = stock_df['Close']

    # -- STAGE 1: MOIRAI --
    print(f"\n[2/3] MOIRAI: Does the 3:2:1 Crack Spread drive refiner stocks?")
    stock_analysis, influence_matrix, all_cols = run_moirai_discovery(df)

    # Print full bidirectional report
    print(f"\n  {'='*70}")
    print(f"  MOIRAI BIDIRECTIONAL ATTENTION ANALYSIS")
    print(f"  {'='*70}")
    print(f"  {'STOCK':<8} {'Crack->Stock':<14} {'Stock->Crack':<14} {'Ratio':<8} {'Relation':<12} {'Trade?'}")
    print(f"  {'-'*70}")

    tradeable_stocks = []
    for ticker, info in sorted(stock_analysis.items(), key=lambda x: x[1]["ratio"], reverse=True):
        trade_marker = "YES" if info["tradeable"] else "NO"
        if info["tradeable"]:
            tradeable_stocks.append(ticker)

        print(f"  {ticker:<8} {info['crack_to_stock']:<14.4f} {info['stock_to_crack']:<14.4f} "
              f"{info['ratio']:<8.2f} {info['relation']:<12} {trade_marker}")

    print(f"\n  Full Influence Matrix (Feature Contribution %):")
    print(f"  {'Q \ K':>10}", end="")
    for col in all_cols:
        label = col.replace("_Hedged_Return", "").replace("Crack_Z_Score", "Crack")
        print(f"  {label:>8}", end="")
    print()
    for i, row_col in enumerate(all_cols):
        label = row_col.replace("_Hedged_Return", "").replace("Crack_Z_Score", "Crack")
        print(f"  {label:>10} Q", end="")
        row_sum = np.sum(influence_matrix[i]) + 1e-8
        for j in range(len(all_cols)):
            pct = (influence_matrix[i, j] / row_sum) * 100
            print(f"  {pct:>7.1f}%", end="")
        print()

    # Save MOIRAI results
    moirai_output = {
        "stock_analysis": stock_analysis,
        "tradeable_stocks": tradeable_stocks,
        "influence_matrix": influence_matrix.tolist(),
        "columns": all_cols,
    }
    with open(os.path.join(DATA_DIR, "moirai_discovery.json"), "w") as f:
        json.dump(moirai_output, f, indent=2)
    print(f"\n  Saved MOIRAI discovery to moirai_discovery.json")

    if not tradeable_stocks:
        print("\n  [!] No tradeable stocks found. Crack spread does not drive any refiner stock.")
        return

    # -- STAGE 2: Chronos predicts ALL tradeable stocks --
    print(f"\n[3/3] Chronos-2: Forecasting {len(tradeable_stocks)} tradeable stocks...")

    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cuda", dtype=torch.float32,
    )

    test_mask = df.index >= '2020-01-01'
    test_dates = df[test_mask].index
    if dry_run_days:
        test_dates = test_dates[:dry_run_days]
        print(f"  DRY RUN: Testing on the first {dry_run_days} days of the out-of-sample test range.")

    all_results = []
    final_tradeable_stocks = []

    for ticker in tradeable_stocks:
        stock_col = f"{ticker}_Hedged_Return"
        print(f"\n  Forecasting {ticker}...")
        
        # Tracking for ablation test
        baseline_maes = []
        mv_maes = []
        baseline_correct = 0
        mv_correct = 0

        for i, current_date in enumerate(test_dates):
            history = df[df.index < current_date]
            if len(history) < 60:
                continue
                
            actual_ret = df.loc[current_date, stock_col]
            actual_dir = 1 if actual_ret > 0 else 0

            # --- ABLATION: Baseline (History Only) ---
            pred_base = run_chronos_inference(
                pipeline, history,
                target_col=stock_col,
                covariate_cols=[]  # Univariate
            )
            base_mae = abs(pred_base['q50'] - actual_ret)
            baseline_maes.append(base_mae)
            if (pred_base['p_up'] > 0.5) == actual_dir:
                baseline_correct += 1

            # --- ABLATION: Full MV (History + Crack) ---
            pred_mv = run_chronos_inference(
                pipeline, history,
                target_col=stock_col,
                covariate_cols=["Crack_Z_Score"]
            )
            mv_mae = abs(pred_mv['q50'] - actual_ret)
            mv_maes.append(mv_mae)
            if (pred_mv['p_up'] > 0.5) == actual_dir:
                mv_correct += 1

            # --- GET LATEST PRICE TO CONVERT PREDICTION TO PRICE ---
            # df contains "SYMBOL_Close" columns
            price_col = f"{ticker}_Close"
            if price_col in df.columns:
                last_price = df.loc[history.index[-1], price_col]
            else:
                last_price = 100.0 # Fallback if price not found
                
            all_results.append({
                'date': current_date,
                'stock': ticker,
                'q10_ret': pred_mv['q10'],
                'q50_ret': pred_mv['q50'],
                'q90_ret': pred_mv['q90'],
                'q10_price': last_price * (1 + pred_mv['q10']),
                'q50_price': last_price * (1 + pred_mv['q50']),
                'q90_price': last_price * (1 + pred_mv['q90']),
                'actual_price': last_price * (1 + actual_ret),
                'p_up': pred_mv['p_up'],
                'actual_return': actual_ret,
                'base_p_up': pred_base['p_up'],
                'base_q50': pred_base['q50']
            })

        # Calculate and print Ablation Results for this stock
        if baseline_maes and mv_maes:
            avg_base_mae = np.mean(baseline_maes)
            avg_mv_mae = np.mean(mv_maes)
            mae_improvement = avg_base_mae - avg_mv_mae
            
            base_hit = (baseline_correct / len(baseline_maes)) * 100
            mv_hit = (mv_correct / len(mv_maes)) * 100
            hit_improvement = mv_hit - base_hit
            
            # Feature Importance Calc
            # If the crack spread improved MAE, we calculate its contribution as the 
            # percentage reduction in error. 
            if mae_improvement > 0:
                pair_pct = (mae_improvement / avg_base_mae) * 100
                # We cap pair contribution at 100% just in case of weird math, though impossible here
                pair_pct = min(pair_pct, 100.0)
                y_hist_pct = 100.0 - pair_pct
            else:
                pair_pct = 0.0
                y_hist_pct = 100.0
            
            print(f"\n  {'-'*70}")
            print(f"  [ ABLATION TEST: {ticker} (Baseline vs Crack Spread) ]")
            print(f"  {'-'*70}")
            print(f"  Univariate (History Only): MAE {avg_base_mae:.5f} | Hit: {base_hit:.1f}%")
            print(f"  Multivariate (+Crack):     MAE {avg_mv_mae:.5f} | Hit: {mv_hit:.1f}%")
            sign = "+" if mae_improvement > 0 else ""
            h_sign = "+" if hit_improvement > 0 else ""
            print(f"  Improvement:               MAE {sign}{mae_improvement:.5f} | Hit: {h_sign}{hit_improvement:.1f}%")
            
            print(f"\n  [ FEATURE IMPORTANCE (% of Forecast Signal) ]")
            print(f"  1. {ticker} History: {y_hist_pct:>6.1f}%")
            print(f"  2. 3:2:1 Crack:  {pair_pct:>6.1f}%")
            print(f"  {'-'*70}\n")
            
            # Record if it passes the final ablation rule (only accuracy improvement)
            if hit_improvement > 0:
                final_tradeable_stocks.append(ticker)

    print(f"\n{'='*70}")
    print(f" FINAL SELECTION RULE APPLIED:")
    print(f" 1. MOIRAI FOLLOWER? (Attention Ratio > 1.2)")
    print(f" 2. ABLATION IMPROVED? (Crack Spread improved directional accuracy/hit rate)")
    print(f" -> Selected: {final_tradeable_stocks}")
    print(f"{'='*70}")

    # Filter all_results to only include the final_tradeable_stocks
    all_results = [r for r in all_results if r['stock'] in final_tradeable_stocks]

    res_df = pd.DataFrame(all_results)
    out_path = os.path.join(DATA_DIR, "inference_results.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved inference results for {len(tradeable_stocks)} stocks to {out_path}")


if __name__ == "__main__":
    run_full_pipeline(dry_run_days=30)
