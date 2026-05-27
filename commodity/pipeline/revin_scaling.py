"""
RevIN Scaling Orchestrator.
Executes the native Snowflake CTAS query to compute the dense temporal grid,
forward fill missing dates, and calculate rolling Z-scores.
Extracts the final reversal dictionary back to the local client.
"""

import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient
from utility.revin_sql import get_revin_sql

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_NORMALIZED"
TARGET_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_READY_MEDIAN"
DICT_OUTPUT_PATH = ROOT / "data" / "reversal_dict.json"

def run() -> None:
    print("=" * 65)
    print("  RevIN SCALING PIPELINE (Native Snowflake)")
    print(f"  Source: {SOURCE_TABLE}")
    print(f"  Target: {TARGET_TABLE}")
    print("=" * 65)
    t0 = time.time()

    with SnowflakeClient() as sf:
        sf.connect()

        # 1. Execute CTAS
        print(f"\n[SQL] Executing Dense Grid & RevIN Window Functions (CTAS)...")
        print("      (This may take several minutes due to the massive Cartesian join)")
        
        sql = get_revin_sql(SOURCE_TABLE, TARGET_TABLE)
        t_sql = time.time()
        
        with sf.cursor() as cur:
            cur.execute(sql)
            print(f"[SQL] Created {TARGET_TABLE} in {time.time()-t_sql:.1f}s")
            
            cur.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
            print(f"[SQL] Dense grid rows generated: {cur.fetchone()[0]:,}")

        # 2. Extract Reversal Dictionary
        print("\n[DICT] Extracting final RevIN parameters for dictionary...")
        dict_query = f"""
            SELECT 
                SYMBOL, 
                ROLLING_MEAN, 
                ROLLING_STD
            FROM {TARGET_TABLE}
            WHERE ROLLING_MEAN IS NOT NULL AND ROLLING_STD IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY SYMBOL ORDER BY ASSESSDATE DESC) = 1
        """
        
        df_dict = sf.read_sql(dict_query)
        
        reversal_dict = {}
        for _, row in df_dict.iterrows():
            reversal_dict[row['SYMBOL']] = {
                "mean": float(row['ROLLING_MEAN']),
                "std": float(row['ROLLING_STD'])
            }
            
        DICT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DICT_OUTPUT_PATH, 'w') as f:
            json.dump(reversal_dict, f, indent=4)
            
        print(f"[DICT] Saved {len(reversal_dict):,} symbols to {DICT_OUTPUT_PATH}")

    print(f"\n[OK] RevIN Scaling Pipeline completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run()
