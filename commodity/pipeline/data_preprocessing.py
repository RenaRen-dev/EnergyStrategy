"""
Centralized Data Pre-Processing for MOIRAI Zero-Shot Forecasting.

Includes Model-Driven Target Discovery.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY"

# Max variates to feed MOIRAI at once (bounded by 8GB VRAM)
# 20 is safe for full 10-day probabilistic forecasting
MAX_VARIATES = 20

# Candidates for the Model-Driven Pre-Scan
MAX_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Step 1: Data Ingestion
# ---------------------------------------------------------------------------
def fetch_data(
    product: str = "Unleaded Gasoline",
    date_from: str = "2020-01-01",
    limit: int = 100000,
    csv_fallback: str | None = None,
) -> pd.DataFrame:
    if csv_fallback:
        print(f"[INGEST] Loading local CSV fallback: {csv_fallback}")
        df = pd.read_csv(csv_fallback)
    else:
        print(f"[INGEST] Fetching from Snowflake: {SOURCE_TABLE}")
        query = f"""
            SELECT SYMBOL, ASSESSDATE, Z_SCORE
            FROM {SOURCE_TABLE}
            WHERE PRODUCT = '{product}'
              AND ASSESSDATE >= '{date_from}'
            ORDER BY SYMBOL, ASSESSDATE
            LIMIT {limit}
        """
        with SnowflakeClient() as sf:
            df = sf.read_sql(query)

    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"])
    print(f"[INGEST] Fetched {len(df):,} rows | {df['SYMBOL'].nunique()} unique symbols")
    return df


# ---------------------------------------------------------------------------
# Step 2: Pivot + Liquidity Filter
# ---------------------------------------------------------------------------
def pivot_and_filter(df: pd.DataFrame, liquidity_threshold: float = 0.80) -> pd.DataFrame:
    print("[PIVOT] Building wide-matrix (Date × Symbol)...")
    pivot_df = df.pivot(index="ASSESSDATE", columns="SYMBOL", values="Z_SCORE")
    pivot_df = pivot_df.asfreq("D")

    valid_mask = pivot_df.notna() & (pivot_df != 0)
    active_symbols = valid_mask.mean()[valid_mask.mean() >= liquidity_threshold].index.tolist()
    pivot_df = pivot_df[active_symbols]
    print(f"[PIVOT] Kept {len(active_symbols)} liquid symbols (>= {liquidity_threshold*100:.0f}% coverage)")

    pivot_df = pivot_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return pivot_df


# ---------------------------------------------------------------------------
# Step 3: Candidate Selection for Model Discovery
# ---------------------------------------------------------------------------
def get_model_candidates(pivot_df: pd.DataFrame, max_candidates: int = MAX_CANDIDATES) -> pd.DataFrame:
    """
    Select top N symbols by volatility to be candidates for the Model-Driven Target Discovery.
    The model will then scan these to find the true "Market Core."
    """
    symbol_variances = pivot_df.var().sort_values(ascending=False)
    candidate_symbols = symbol_variances.head(max_candidates).index.tolist()
    print(f"[CANDIDATES] Selected top {len(candidate_symbols)} volatile symbols for Model-Driven Discovery.")
    return pivot_df[candidate_symbols].copy()


# ---------------------------------------------------------------------------
# Step 4: Finalize Multivariate Matrix (Post-Discovery)
# ---------------------------------------------------------------------------
def finalize_multivariate_matrix(pivot_df: pd.DataFrame, discovered_target: str, max_variates: int = MAX_VARIATES) -> dict:
    """
    After the model has discovered the target, we find the most correlated
    peers to THAT target to build the final inference matrix.
    """
    print(f"[FINALIZE] Building final matrix around discovered target: {discovered_target}...")
    corr_matrix = pivot_df.corr()
    target_corr = corr_matrix[discovered_target].abs().sort_values(ascending=False)
    
    selected_symbols = target_corr.head(max_variates).index.tolist()
    if discovered_target in selected_symbols:
        selected_symbols.remove(discovered_target)
    selected_symbols = [discovered_target] + selected_symbols
    selected_symbols = selected_symbols[:max_variates]

    selected_df = pivot_df[selected_symbols].copy()
    selected_df.index = pd.DatetimeIndex(selected_df.index, freq="D")

    return {
        "multivariate_df": selected_df,
        "target_columns": selected_symbols,
        "target_symbol": discovered_target,
        "correlations": corr_matrix[discovered_target][selected_symbols]
    }


# ---------------------------------------------------------------------------
# Full Pipeline Entrypoint
# ---------------------------------------------------------------------------
def run_preprocessing(
    product: str = "Unleaded Gasoline",
    date_from: str = "2020-01-01",
    limit: int = 100000,
    csv_fallback: str | None = None,
    liquidity_threshold: float = 0.80,
) -> dict:
    print("=" * 65)
    print("  MOIRAI DATA PREPROCESSING PIPELINE")
    print("=" * 65)

    df = fetch_data(product=product, date_from=date_from, limit=limit, csv_fallback=csv_fallback)
    pivot_df = pivot_and_filter(df, liquidity_threshold=liquidity_threshold)
    candidate_df = get_model_candidates(pivot_df)

    return {
        "pivot_df": pivot_df,
        "candidate_df": candidate_df,
        "candidate_columns": list(candidate_df.columns),
    }
