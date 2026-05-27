"""
Data Pre-processing Pipeline — single entry point.

Runs entirely in Snowflake:
  1. Registers a Python UDF (parse_description) in Snowflake
  2. Creates PRICEDATA_PARSED table via CREATE TABLE AS SELECT

No data is downloaded locally.

Usage:
    python -m pipeline.data_preprocessing                 # full 163M rows
    python -m pipeline.data_preprocessing --limit 10000   # test subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient
from utility.parse_description_udf import get_udf_sql, get_ctas_sql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_TABLE = "SPGE_MARKETDATA_SHARE.MDV2.PRICEDATA"
TARGET_DB = "CMDTYA"
TARGET_SCHEMA = "PUBLIC"
TARGET_TABLE = "PRICEDATA_PARSED"
TARGET_FQN = f"{TARGET_DB}.{TARGET_SCHEMA}.{TARGET_TABLE}"
UDF_FQN = f"{TARGET_DB}.{TARGET_SCHEMA}.PARSE_DESCRIPTION"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_register_udf(sf: SnowflakeClient) -> None:
    """Register the parse_description Python UDF in Snowflake."""
    print("[STEP 1/3] Registering Python UDF in Snowflake...")
    t0 = time.time()
    udf_sql = get_udf_sql(UDF_FQN)
    with sf.cursor() as cur:
        cur.execute(udf_sql)
    print(f"[OK]      UDF registered: {UDF_FQN} ({time.time()-t0:.1f}s)")


def step_create_table(sf: SnowflakeClient, *, limit: int | None = None) -> None:
    """Create PRICEDATA_PARSED via CREATE TABLE AS SELECT (server-side)."""
    row_info = f" (LIMIT {limit:,})" if limit else " (full table)"
    print(f"[STEP 2/3] Creating {TARGET_FQN}{row_info}...")
    print("           All processing runs on Snowflake compute.")

    ctas_sql = get_ctas_sql(
        udf_fqn=UDF_FQN,
        source_table=SOURCE_TABLE,
        target_fqn=TARGET_FQN,
        limit=limit,
    )

    t0 = time.time()
    with sf.cursor() as cur:
        cur.execute(ctas_sql)
    print(f"[OK]      Table created in {time.time()-t0:.1f}s")


def step_verify(sf: SnowflakeClient) -> None:
    """Verify the created table and show sample rows."""
    print(f"[STEP 3/3] Verifying {TARGET_FQN}...")

    with sf.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TARGET_FQN}")
        count = cur.fetchone()[0]
        print(f"[OK]      {count:,} rows in {TARGET_FQN}")

        # IS_SPOT distribution
        cur.execute(f"""
            SELECT IS_SPOT, COUNT(*) AS cnt
            FROM {TARGET_FQN}
            GROUP BY IS_SPOT
            ORDER BY cnt DESC
        """)
        print(f"\n           IS_SPOT distribution:")
        for row in cur.fetchall():
            print(f"             {row[0]}: {row[1]:,}")

        # Delivery distribution
        cur.execute(f"""
            SELECT DELIVERY, COUNT(*) AS cnt
            FROM {TARGET_FQN}
            GROUP BY DELIVERY
            ORDER BY cnt DESC
            LIMIT 10
        """)
        print(f"\n           Delivery distribution (top 10):")
        for row in cur.fetchall():
            print(f"             {row[0]:35s} {row[1]:>12,}")

        # Sample rows
        cur.execute(f"""
            SELECT DESCRIPTION, PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING, IS_SPOT
            FROM {TARGET_FQN}
            LIMIT 5
        """)
        cols = [d[0] for d in cur.description]
        print(f"\n           Sample rows:")
        for row in cur.fetchall():
            print(f"             {dict(zip(cols, row))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(*, limit: int | None = None) -> None:
    """Execute the full data pre-processing pipeline."""
    print("=" * 65)
    print("  DATA PRE-PROCESSING PIPELINE")
    print(f"  Source: {SOURCE_TABLE}")
    print(f"  Target: {TARGET_FQN}")
    print("=" * 65)
    t0 = time.time()

    with SnowflakeClient() as sf:
        sf.connect()

        # Set database/schema context
        with sf.cursor() as cur:
            cur.execute(f"USE DATABASE {TARGET_DB}")
            cur.execute(f"USE SCHEMA {TARGET_SCHEMA}")

        step_register_udf(sf)
        step_create_table(sf, limit=limit)
        step_verify(sf)

    total = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  PIPELINE COMPLETE — {total:.1f}s total")
    print(f"{'=' * 65}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the data pre-processing pipeline"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit source rows (default: all ~163M rows)",
    )
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
