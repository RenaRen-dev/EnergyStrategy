"""
Snowflake Aggregation Pipeline.

Creates a high-quality aggregated table by grouping by Date and Symbol
to calculate the Daily Median Price. This reduces granularity and
improves signal-to-noise ratio for the ML models.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_READY"
TARGET_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY"

def create_aggregated_table():
    sql = f"""
    CREATE OR REPLACE TABLE {TARGET_TABLE} AS
    SELECT 
        SYMBOL, 
        ASSESSDATE, 
        MEDIAN(NORMALIZED_VALUE_USD_GAL) AS MEDIAN_PRICE,
        MEDIAN(Z_SCORE) AS Z_SCORE,
        ANY_VALUE(PRODUCT) AS PRODUCT,
        ANY_VALUE(GRADE) AS GRADE,
        ANY_VALUE(GEOGRAPHY) AS GEOGRAPHY,
        ANY_VALUE(DELIVERY) AS DELIVERY,
        ANY_VALUE(TIMING) AS TIMING
    FROM {SOURCE_TABLE}
    GROUP BY SYMBOL, ASSESSDATE
    ORDER BY SYMBOL, ASSESSDATE;
    """
    
    print(f"Creating aggregated table {TARGET_TABLE} from {SOURCE_TABLE}...")
    with SnowflakeClient() as sf:
        sf.connect()
        sf.execute(sql)
    print("Aggregation complete.")

if __name__ == "__main__":
    create_aggregated_table()
