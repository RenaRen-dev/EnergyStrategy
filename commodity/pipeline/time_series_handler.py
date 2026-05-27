"""
Time-Series Data Handler for Energy Commodities.

This script pulls a subset of physically normalized data from Snowflake,
aligns the time-series grid to a continuous daily frequency (forward-filling weekends),
computes RevIN (Reversible Instance Normalization) parameters (Rolling Mean/Std),
and extracts a reversal dictionary to reconstruct absolute prices from Z-scores.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_NORMALIZED"
DICT_OUTPUT_PATH = ROOT / "data" / "reversal_dict.json"

def fetch_data(product: str = 'Gas Oil', limit: int | None = 100000) -> pd.DataFrame:
    """
    Pulls a subset of the normalized data from Snowflake. Use limit=None to fetch all.
    """
    limit_str = f"LIMIT {limit}" if limit else ""
    where_clause = f"WHERE PRODUCT = '{product}'" if product != "ALL" else ""
    
    print(f"Fetching data for PRODUCT = '{product}' ({'All Rows' if not limit else limit_str})...")
    
    query = f"""
        SELECT 
            SYMBOL, ASSESSDATE, NORMALIZED_VALUE_USD_GAL,
            PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING
        FROM {SOURCE_TABLE}
        {where_clause}
        ORDER BY SYMBOL, ASSESSDATE
        {limit_str}
    """
    
    with SnowflakeClient() as sf:
        df = sf.read_sql(query)
        
    print(f"Fetched {len(df)} rows from Snowflake.")
    return df

def align_and_scale_timeseries(df: pd.DataFrame, window_days: int = 256) -> pd.DataFrame:
    """
    Converts dates, resamples to a strict daily grid (forward-filling weekends),
    and computes the RevIN parameters (rolling mean and standard deviation) 
    along with the safely bounded Z-Score.
    """
    print("Aligning temporal grid and computing RevIN parameters...")
    
    # 1. Convert to datetime
    df['ASSESSDATE'] = pd.to_datetime(df['ASSESSDATE'])
    
    # Static categorical columns to forward fill along with the price
    static_cols = ['PRODUCT', 'GRADE', 'GEOGRAPHY', 'DELIVERY', 'TIMING']
    
    # 2. Group by SYMBOL and resample to daily frequency
    def process_group(group):
        # Handle potential duplicate dates for the same symbol (e.g., intra-day quotes)
        # by taking the mean of the price and keeping the first instance of categoricals.
        # This also automatically sets the index to ASSESSDATE.
        group = group.groupby('ASSESSDATE').agg({
            'NORMALIZED_VALUE_USD_GAL': 'mean',
            'PRODUCT': 'first',
            'GRADE': 'first',
            'GEOGRAPHY': 'first',
            'DELIVERY': 'first',
            'TIMING': 'first'
        })
        # Resample to daily frequency and forward fill
        resampled = group.resample('D').ffill()
        
        # 3. Compute RevIN parameters using a rolling window
        # We use min_periods=1 to start producing Z-scores immediately if desired, 
        # or leave it to default (window_days) to wait for a full window.
        resampled['ROLLING_MEAN'] = resampled['NORMALIZED_VALUE_USD_GAL'].rolling(window=window_days, min_periods=1).mean()
        resampled['ROLLING_STD'] = resampled['NORMALIZED_VALUE_USD_GAL'].rolling(window=window_days, min_periods=1).std()
        
        # Add an epsilon safety floor to prevent division by zero
        epsilon = 1e-8
        
        # 4. Compute safely bounded Z-Score
        resampled['Z_SCORE'] = (resampled['NORMALIZED_VALUE_USD_GAL'] - resampled['ROLLING_MEAN']) / (resampled['ROLLING_STD'] + epsilon)
        
        # Clip Z-score to [-3, 3] to handle extreme outliers safely in the ML model
        resampled['Z_SCORE'] = resampled['Z_SCORE'].clip(lower=-3.0, upper=3.0)
        
        return resampled

    # Apply processing per symbol and reset index
    processed_df = df.groupby('SYMBOL').apply(process_group, include_groups=False)
    processed_df = processed_df.reset_index() # Bring ASSESSDATE back as a column
    
    print(f"Grid alignment complete. Expanded to {len(processed_df)} continuous daily rows.")
    return processed_df

def extract_reversal_dictionary(df: pd.DataFrame, output_path: Path) -> dict:
    """
    Extracts the most recent non-null ROLLING_MEAN and ROLLING_STD for each unique SYMBOL
    and saves the mapping as a local JSON file.
    """
    print(f"Extracting RevIN reversal dictionary...")
    
    # Get the last row for each symbol (which contains the most recent rolling stats)
    # dropna is used to ensure we have valid rolling stats
    latest_stats = df.dropna(subset=['ROLLING_MEAN', 'ROLLING_STD']).groupby('SYMBOL').last()
    
    reversal_dict = {}
    for symbol, row in latest_stats.iterrows():
        reversal_dict[symbol] = {
            "mean": float(row['ROLLING_MEAN']),
            "std": float(row['ROLLING_STD'])
        }
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(reversal_dict, f, indent=4)
        
    print(f"Saved reversal dictionary for {len(reversal_dict)} symbols to {output_path}")
    return reversal_dict

def main():
    parser = argparse.ArgumentParser(description="Time-Series Data Handler (RevIN Processing)")
    parser.add_argument("--limit", type=int, default=100000, help="Row limit for the query (default: 100000). Set to 0 to fetch all.")
    parser.add_argument("--product", type=str, default="ALL", help="Specific product to filter by, or 'ALL' for the entire dataset.")
    args = parser.parse_args()
    
    limit_val = args.limit if args.limit > 0 else None

    print("=" * 65)
    print("  TIME-SERIES DATA HANDLER (RevIN Processing)")
    print("=" * 65)
    
    # 1. Data Ingestion
    df_raw = fetch_data(product=args.product, limit=limit_val)
    
    if df_raw.empty:
        print("[ERROR] No data fetched. Exiting.")
        return
        
    # 2. Temporal Grid Alignment & RevIN Computation
    df_processed = align_and_scale_timeseries(df_raw, window_days=256)
    
    # 3. RevIN Reversal Dictionary
    reversal_dict = extract_reversal_dictionary(df_processed, DICT_OUTPUT_PATH)
    
    print("=" * 65)
    print("  DATA HANDLING COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()
