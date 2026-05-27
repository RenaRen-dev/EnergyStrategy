"""
Normalization step of the ELT Pipeline.
Handles FX historical data ingestion and the deterministic
physical/financial normalization CTAS across the 163M row dataset.

Output: All values unified to USD per US Gallon (USD/GAL).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient
from utility.fx_client import ingest_historical_fx
from utility.normalization_sql import get_physical_normalization_sql

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_PARSED"
TARGET_TABLE = "CMDTYA.PUBLIC.PRICEDATA_NORMALIZED"


def run(*, skip_fx_ingest: bool = False) -> None:
    print("=" * 65)
    print("  NORMALIZATION PIPELINE")
    print(f"  Source: {SOURCE_TABLE}")
    print(f"  Target: {TARGET_TABLE}")
    print("=" * 65)
    t0 = time.time()

    with SnowflakeClient() as sf:
        sf.connect()

        # 1. Fetch Date Range
        with sf.cursor() as cur:
            cur.execute(f"SELECT MIN(ASSESSDATE), MAX(ASSESSDATE) FROM {SOURCE_TABLE}")
            min_date, max_date = cur.fetchone()

        if not min_date or not max_date:
            print("[ERROR] No data found in parsed table.")
            return

        # 2. Ingest FX Data
        if not skip_fx_ingest:
            success = ingest_historical_fx(sf, min_date, max_date)
            if not success:
                print("[ERROR] FX Ingestion failed. Aborting Normalization.")
                return
        else:
            print("[FX] Skipping FX ingestion as requested.")

        # 3. Physical & Financial Normalization
        print("\n[SQL] Executing Physical & Financial Normalization CTAS...")
        sql = get_physical_normalization_sql(SOURCE_TABLE, TARGET_TABLE)

        t_sql = time.time()
        with sf.cursor() as cur:
            cur.execute(sql)
            print(f"[SQL] Created {TARGET_TABLE} in {time.time()-t_sql:.1f}s")

            # Verify results
            cur.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
            print(f"[SQL] Total rows processed: {cur.fetchone()[0]:,}")

    print(f"\n[OK] Normalization Pipeline completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
