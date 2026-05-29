"""Download SPGE_MARKETDATA_SHARE.MDV2.PRICEDATA from Snowflake.

Three subcommands so you don't accidentally pull 163M rows on the first run:

  test     -> verify connection works (1 query, instant)
  inspect  -> show schema, row count, date range, sample rows
               (4 small queries, ~30 sec)
  download -> chunked pull into per-period parquet files
               (resumable; skips chunks already on disk)

Usage:
  cd /Users/rena/Downloads/Energy-Strategy-main-2
  source .venv/bin/activate                       # or whatever venv you use
  pip install -r requirements.txt                 # one-time

  python platts/scripts/01_download_pricedata.py test
  python platts/scripts/01_download_pricedata.py inspect
  python platts/scripts/01_download_pricedata.py download \\
      --start 2014-01-01 --end 2026-05-01 --chunk-months 6

Resumable: each chunk writes to platts/outputs/pricedata/raw/
chunk_YYYY-MM-DD_YYYY-MM-DD.parquet. Re-run = skip existing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env from the project root BEFORE importing SnowflakeClient,
# because SnowflakeClient resolves env vars at import.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from commodity.utility.snowflake_client import SnowflakeClient  # noqa: E402

SOURCE_TABLE = "SPGE_MARKETDATA_SHARE.MDV2.PRICEDATA"
OUT_DIR = PROJECT_ROOT / "platts" / "outputs" / "pricedata" / "raw"


def _activate_session(sf: SnowflakeClient) -> None:
    """Explicitly USE warehouse/role/database/schema. The Snowflake connector's
    connection params get ignored on some accounts (warehouse 57P03 error),
    so we send the USE commands directly.
    """
    import os
    role = os.getenv("SNOWFLAKE_ROLE")
    wh = os.getenv("SNOWFLAKE_WAREHOUSE")
    db = os.getenv("SNOWFLAKE_DATABASE")
    sch = os.getenv("SNOWFLAKE_SCHEMA")
    with sf.cursor() as cur:
        if role:
            cur.execute(f'USE ROLE "{role}"')
        if wh:
            # Try as-is first (preserves case if user quoted it),
            # then uppercase as Snowflake's default normalization.
            try:
                cur.execute(f'USE WAREHOUSE "{wh}"')
            except Exception:
                cur.execute(f'USE WAREHOUSE {wh.upper()}')
        if db:
            cur.execute(f'USE DATABASE "{db}"')
        if sch:
            cur.execute(f'USE SCHEMA "{sch}"')
        cur.execute(
            "SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        )
        r, w, d, s = cur.fetchone()
    print(f"  Session: role={r}, wh={w}, db={d}, schema={s}")


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------

def cmd_test() -> int:
    """Connect, activate warehouse/db/schema, and list available warehouses."""
    print("=" * 65)
    print("  CONNECTION TEST")
    print("=" * 65)
    try:
        with SnowflakeClient() as sf:
            _activate_session(sf)
            print("\n  Available warehouses you can use:")
            wh_df = sf.read_sql("SHOW WAREHOUSES")
            for _, row in wh_df.iterrows():
                state = row.get("state", row.get("STATE", "?"))
                size = row.get("size", row.get("SIZE", "?"))
                print(f"    - {row['name']:<25} state={state}  size={size}")
        print("\n[OK] Connection + session activation successful.")
        return 0
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        print("\nCommon causes:")
        print("  - SNOWFLAKE_WAREHOUSE name wrong (run SHOW WAREHOUSES in Snowsight)")
        print("  - Private key passphrase missing/wrong in .env")
        print("  - Warehouse not running / role not granted")
        return 1


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------

def cmd_inspect() -> int:
    """Show schema, row count, date range, sample rows. Read-only, fast."""
    print("=" * 65)
    print(f"  INSPECT  {SOURCE_TABLE}")
    print("=" * 65)
    with SnowflakeClient() as sf:
        _activate_session(sf)
        # 1. Schema
        print("\n[1/4] Columns:")
        cols_df = sf.read_sql(f"DESCRIBE TABLE {SOURCE_TABLE}")
        for _, row in cols_df.iterrows():
            print(f"      {row['name']:<25} {row['type']:<25} null={row['null?']}")

        # 2. Row count
        print(f"\n[2/4] Row count (this may take 30-60 sec on 163M rows)...")
        t0 = time.time()
        count_df = sf.read_sql(f"SELECT COUNT(*) AS n FROM {SOURCE_TABLE}")
        n_rows = int(count_df["N"].iloc[0])
        print(f"      Total rows: {n_rows:,}  ({time.time()-t0:.1f}s)")

        # 3. Date range (assumes ASSESSDATE column; falls back to first DATE col)
        date_col = _detect_date_col(cols_df)
        print(f"\n[3/4] Date range on column {date_col!r}:")
        rng_df = sf.read_sql(
            f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d FROM {SOURCE_TABLE}"
        )
        print(f"      MIN({date_col}) = {rng_df['MIN_D'].iloc[0]}")
        print(f"      MAX({date_col}) = {rng_df['MAX_D'].iloc[0]}")

        # 4. Sample
        print(f"\n[4/4] First 5 rows:")
        sample = sf.read_sql(f"SELECT * FROM {SOURCE_TABLE} LIMIT 5")
        with __import__("pandas").option_context(
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 60,
        ):
            print(sample)

    print("\n[OK] Inspection complete. Use 'download' next.")
    return 0


def _detect_date_col(cols_df) -> str:
    """Pick the date column to filter on. Prefer ASSESSDATE, fall back to first DATE column."""
    names = [n.upper() for n in cols_df["name"]]
    if "ASSESSDATE" in names:
        return "ASSESSDATE"
    for _, row in cols_df.iterrows():
        if "DATE" in row["type"].upper() or "TIMESTAMP" in row["type"].upper():
            return row["name"]
    raise RuntimeError("No date/timestamp column found in PRICEDATA")


# ---------------------------------------------------------------------------
# Subcommand: download
# ---------------------------------------------------------------------------

def cmd_download(start: str, end: str, chunk_months: int) -> int:
    """Pull rows in [start, end) in chunk_months-wide windows, save to parquet."""
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _chunked_date_ranges(start, end, chunk_months)
    print("=" * 65)
    print(f"  DOWNLOAD  {SOURCE_TABLE}")
    print(f"  Range:    {start} to {end}")
    print(f"  Chunks:   {len(chunks)} x {chunk_months}-month windows")
    print(f"  Output:   {OUT_DIR}")
    print("=" * 65)

    with SnowflakeClient() as sf:
        _activate_session(sf)
        # Detect date column once
        cols_df = sf.read_sql(f"DESCRIBE TABLE {SOURCE_TABLE}")
        date_col = _detect_date_col(cols_df)
        print(f"  Filter column: {date_col}\n")

        total_rows = 0
        total_secs = 0.0
        for i, (c_start, c_end) in enumerate(chunks, 1):
            fname = f"chunk_{c_start}_{c_end}.parquet"
            out_path = OUT_DIR / fname

            if out_path.exists():
                size_mb = out_path.stat().st_size / 1e6
                print(f"  [{i:>3}/{len(chunks)}] SKIP   {fname}  ({size_mb:.1f} MB on disk)")
                continue

            sql = f"""
                SELECT *
                FROM {SOURCE_TABLE}
                WHERE {date_col} >= '{c_start}'
                  AND {date_col} <  '{c_end}'
            """
            t0 = time.time()
            df = sf.read_sql(sql)
            elapsed = time.time() - t0

            df.to_parquet(out_path, compression="snappy", index=False)
            size_mb = out_path.stat().st_size / 1e6

            total_rows += len(df)
            total_secs += elapsed
            print(
                f"  [{i:>3}/{len(chunks)}] OK     {fname}  "
                f"{len(df):>10,} rows  "
                f"{size_mb:>6.1f} MB  "
                f"{elapsed:>6.1f}s"
            )

    print("\n" + "=" * 65)
    print(f"  TOTAL new rows downloaded: {total_rows:,}")
    print(f"  TOTAL download time:        {total_secs/60:.1f} min")
    print(f"  Files in {OUT_DIR}")
    return 0


def cmd_verify() -> int:
    """Compare local chunks against Snowflake source: counts, date ranges, samples."""
    import pandas as pd

    files = sorted(OUT_DIR.glob("chunk_*.parquet"))
    if not files:
        print(f"[FAIL] No chunks found in {OUT_DIR}")
        return 1

    print("=" * 65)
    print(f"  VERIFY  ({len(files)} chunks in {OUT_DIR})")
    print("=" * 65)

    # --- Local stats: per-chunk count, min/max date, total ---
    print("\n[1/3] Local chunk audit:")
    print(f"  {'chunk':<46} {'rows':>12}  {'min date':<11}  {'max date':<11}")
    print(f"  {'-'*46} {'-'*12}  {'-'*11}  {'-'*11}")
    total_rows = 0
    overall_min = None
    overall_max = None
    for f in files:
        df = pd.read_parquet(f, columns=["ASSESSDATE"])
        n = len(df)
        mn = df["ASSESSDATE"].min()
        mx = df["ASSESSDATE"].max()
        total_rows += n
        overall_min = mn if overall_min is None or mn < overall_min else overall_min
        overall_max = mx if overall_max is None or mx > overall_max else overall_max
        print(f"  {f.name:<46} {n:>12,}  {str(mn)[:10]:<11}  {str(mx)[:10]:<11}")
    print(f"  {'TOTAL':<46} {total_rows:>12,}  {str(overall_min)[:10]:<11}  {str(overall_max)[:10]:<11}")

    # --- Cross-check against Snowflake for the same range ---
    sf_start = str(overall_min)[:10]
    sf_end = str(overall_max)[:10]
    print(f"\n[2/3] Snowflake source COUNT for [{sf_start}, {sf_end}]:")
    with SnowflakeClient() as sf:
        _activate_session(sf)
        sf_count = int(sf.read_sql(
            f"SELECT COUNT(*) AS n FROM {SOURCE_TABLE} "
            f"WHERE ASSESSDATE >= '{sf_start}' AND ASSESSDATE <= '{sf_end}'"
        )["N"].iloc[0])
        sf_min_max = sf.read_sql(
            f"SELECT MIN(ASSESSDATE) AS mn, MAX(ASSESSDATE) AS mx FROM {SOURCE_TABLE} "
            f"WHERE ASSESSDATE >= '{sf_start}' AND ASSESSDATE <= '{sf_end}'"
        )

    print(f"  Snowflake rows in range: {sf_count:,}")
    print(f"  Local rows in range:     {total_rows:,}")
    diff = sf_count - total_rows
    pct = 100 * diff / sf_count if sf_count else 0
    status = "OK" if abs(pct) < 0.01 else ("WARN" if abs(pct) < 1 else "FAIL")
    print(f"  Diff: {diff:+,} ({pct:+.4f}%)  -> {status}")
    print(f"  Snowflake MIN={sf_min_max['MN'].iloc[0]}  MAX={sf_min_max['MX'].iloc[0]}")

    # --- Sample rows from one chunk ---
    print(f"\n[3/3] Sample rows from {files[0].name}:")
    sample = pd.read_parquet(files[0]).head(5)
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_colwidth", 50):
        print(sample)

    print(f"\n  Unique SYMBOLs across all chunks (this scans every chunk briefly)...")
    all_symbols = set()
    for f in files:
        syms = pd.read_parquet(f, columns=["SYMBOL"])["SYMBOL"].dropna().unique()
        all_symbols.update(syms)
    print(f"  Distinct symbols: {len(all_symbols):,}")

    return 0


def _chunked_date_ranges(start: str, end: str, months: int) -> list[tuple[str, str]]:
    """Split [start, end) into windows of `months` calendar months."""
    import pandas as pd
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks = []
    cur = start_ts
    while cur < end_ts:
        nxt = cur + pd.DateOffset(months=months)
        if nxt > end_ts:
            nxt = end_ts
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("test", help="Verify Snowflake connection")
    sub.add_parser("inspect", help="Show schema, row count, date range, sample")
    sub.add_parser("verify", help="Audit downloaded chunks vs Snowflake source")

    dl = sub.add_parser("download", help="Chunked download to parquet")
    dl.add_argument("--start", required=True, help="ISO date, inclusive (e.g. 2014-01-01)")
    dl.add_argument("--end", required=True, help="ISO date, exclusive (e.g. 2026-05-01)")
    dl.add_argument("--chunk-months", type=int, default=6, help="Months per chunk (default 6)")

    args = p.parse_args()
    if args.cmd == "test":
        return cmd_test()
    if args.cmd == "inspect":
        return cmd_inspect()
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "download":
        return cmd_download(args.start, args.end, args.chunk_months)
    return 2


if __name__ == "__main__":
    sys.exit(main())
