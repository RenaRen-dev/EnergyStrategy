"""
Download PRICEDATA from Snowflake and optionally save as CSV.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

TABLE = "CMDTYA.PUBLIC.PRICEDATA_PARSED"
OUTPUT_DIR = ROOT / "data"

BATCH_SIZE = 500_000  # rows per fetch batch


def download_data(*, limit: int | None = None, to_csv: bool = False) -> Path:
    """Download PRICEDATA from Snowflake and save as Parquet (and optionally CSV).

    Args:
        limit: If set, download only this many rows.
        to_csv: If True, also save a CSV version.

    Returns:
        Path to the saved file (Parquet by default, or CSV if only CSV was requested).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_file = OUTPUT_DIR / "pricedata.parquet"

    sql = f"SELECT * FROM {TABLE}"
    if limit:
        sql += f" LIMIT {int(limit)}"

    print(f"[INFO] Querying: {sql}")
    t0 = time.time()

    with SnowflakeClient() as sf:
        with sf.cursor() as cur:
            cur.execute(sql)

            batches: list[pd.DataFrame] = []
            batch_num = 0

            while True:
                rows = cur.fetchmany(BATCH_SIZE)
                if not rows:
                    break

                cols = [desc[0] for desc in cur.description]
                batch_df = pd.DataFrame(rows, columns=cols)
                batches.append(batch_df)
                batch_num += 1

                total_rows = sum(len(b) for b in batches)
                elapsed = time.time() - t0
                print(
                    f"  Batch {batch_num}: {len(batch_df):,} rows "
                    f"({total_rows:,} total, {elapsed:.1f}s elapsed)"
                )

    if not batches:
        print("[WARN] No rows returned.")
        return parquet_file

    df = pd.concat(batches, ignore_index=True)
    elapsed = time.time() - t0
    print(f"[INFO] Downloaded {len(df):,} rows in {elapsed:.1f}s. Saving to Parquet...")

    df.to_parquet(parquet_file, index=False, engine="pyarrow")
    print(f"[OK] Saved to {parquet_file}")

    if to_csv:
        csv_file = parquet_file.with_suffix(".csv")
        print(f"[INFO] Converting to CSV...")
        df.to_csv(csv_file, index=False)
        print(f"[OK] Saved to {csv_file}")
        return csv_file

    return parquet_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PRICEDATA from Snowflake")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of rows to download",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also save the data as a CSV file",
    )
    args = parser.parse_args()
    download_data(limit=args.limit, to_csv=args.csv)


if __name__ == "__main__":
    main()
