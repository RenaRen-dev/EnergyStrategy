"""Local pandas port of the commodity Snowflake pipeline.

Reads raw chunks from platts/outputs/pricedata/raw/ (never modified) and
writes processed outputs alongside:

    raw/                     <- source chunks (untouched)
    parsed/                  <- + PRODUCT/GRADE/GEOGRAPHY/DELIVERY/TIMING/IS_SPOT
    normalized/              <- + NORMALIZED_VALUE_USD_GAL
    fx_rates.parquet         <- cached FRED CAD/EUR rates
    zscore/
      pricedata_ml_ready.parquet   <- per-symbol 256-day rolling Z

Subcommands:
    parse     raw/*.parquet      -> parsed/*.parquet           (chunk-by-chunk)
    fx        download FRED      -> fx_rates.parquet           (one-shot)
    normalize parsed/*.parquet   -> normalized/*.parquet       (chunk-by-chunk)
    zscore    normalized/*.parq  -> zscore/pricedata_ml_ready  (monolithic)
    all       run the above four sequentially

Every stage is resumable: re-running skips outputs already on disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Reuse the commodity DESCRIPTION parser — it's pure Python, no Snowflake needed.
from commodity.utility.parse_description_udf import parse_description_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = PROJECT_ROOT / "platts" / "outputs" / "pricedata"
RAW_DIR      = BASE_DIR / "raw"
PARSED_DIR   = BASE_DIR / "parsed"
NORM_DIR     = BASE_DIR / "normalized"
ZSCORE_DIR   = BASE_DIR / "zscore"
FX_PATH      = BASE_DIR / "fx_rates.parquet"

ZSCORE_OUT   = ZSCORE_DIR / "pricedata_ml_ready.parquet"

# Specific gravities (BBL per metric ton) — see commodity/utility/normalization_sql.py
SPECIFIC_GRAVITY = {
    "lighter": 8.5,   # Gasoline, Naphtha, RBOB
    "medium":  7.45,  # Diesel, Gas Oil, Kerosene, Jet, Stove Oil
    "heavy":   6.3,   # Heavy Fuel Oil, Bunker, Furnace Oil
    "generic": 7.0,
}

# Rolling window — matches commodity/utility/revin_sql.py
ROLLING_WINDOW = 256


# ---------------------------------------------------------------------------
# STAGE 1 — parse DESCRIPTION
# ---------------------------------------------------------------------------

def cmd_parse() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("chunk_*.parquet"))
    if not files:
        print(f"[FAIL] No raw chunks in {RAW_DIR}")
        return 1
    print(f"[PARSE] {len(files)} chunks -> {PARSED_DIR}")

    for i, f in enumerate(files, 1):
        out_path = PARSED_DIR / f.name
        if out_path.exists():
            print(f"  [{i:>3}/{len(files)}] SKIP   {f.name}")
            continue

        t0 = time.time()
        df = pd.read_parquet(f)
        parsed = parse_description_batch(df["DESCRIPTION"])
        # parse_description_batch returns Series of dicts; expand to columns.
        # Uppercase keys to match Snowflake convention (Product -> PRODUCT, etc).
        parsed_df = pd.DataFrame(list(parsed.values), index=parsed.index)
        parsed_df.columns = [c.upper() for c in parsed_df.columns]
        df = df.join(parsed_df)
        df.to_parquet(out_path, compression="snappy", index=False)
        print(f"  [{i:>3}/{len(files)}] OK     {f.name}  {len(df):>10,} rows  {time.time()-t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# STAGE 2a — download FRED FX rates
# ---------------------------------------------------------------------------

def cmd_fx(force: bool = False) -> int:
    """Download DEXCAUS, DEXUSEU, EXGEUS from FRED and produce a daily CAD/EUR table."""
    if FX_PATH.exists() and not force:
        print(f"[FX] Already cached: {FX_PATH}")
        return 0

    FX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[FX] Downloading from FRED...")

    def fetch(sid: str) -> pd.Series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        df = pd.read_csv(url, index_col="observation_date", parse_dates=True, na_values=".")
        return df[sid]

    cad = fetch("DEXCAUS")  # CAD per USD
    eur = fetch("DEXUSEU")  # USD per EUR
    dem = fetch("EXGEUS")   # DEM per USD (monthly, used for pre-1999 EUR synthesis)

    # Detect chunk date range so we cover everything we have on disk
    raw_files = sorted(RAW_DIR.glob("chunk_*.parquet"))
    if raw_files:
        min_d = pd.to_datetime("1995-01-01")
        max_d = pd.to_datetime("2030-01-01")
    else:
        min_d, max_d = pd.Timestamp("1995-01-01"), pd.Timestamp("2030-01-01")

    grid = pd.date_range(min_d, max_d, freq="D")
    cad = cad.reindex(grid).ffill().bfill()
    eur = eur.reindex(grid).ffill()
    dem = dem.reindex(grid).ffill().bfill()

    # Synthetic Euro for pre-1999: 1.95583 DEM/EUR
    synth_eur = 1.95583 / dem
    eur_to_usd = eur.combine_first(synth_eur)
    cad_to_usd = 1.0 / cad

    fx = pd.DataFrame({
        "DATE": grid,
        "CAD_TO_USD": cad_to_usd.values,
        "EUR_TO_USD": eur_to_usd.values,
    }).dropna()
    fx["DATE"] = pd.to_datetime(fx["DATE"]).dt.date

    fx.to_parquet(FX_PATH, compression="snappy", index=False)
    print(f"[FX] Saved {len(fx):,} daily rows -> {FX_PATH}")
    return 0


# ---------------------------------------------------------------------------
# STAGE 2b — currency + UOM normalization
# ---------------------------------------------------------------------------

def _bbl_per_mt(product: str | None) -> float:
    """Specific-gravity lookup based on PRODUCT string. Matches commodity SQL."""
    if not isinstance(product, str):
        return SPECIFIC_GRAVITY["generic"]
    p = product.lower()
    if "gasoline" in p or "naphtha" in p or "rbob" in p:
        return SPECIFIC_GRAVITY["lighter"]
    if "diesel" in p or "gas oil" in p or "kerosene" in p or "jet" in p or "stove oil" in p:
        return SPECIFIC_GRAVITY["medium"]
    if "heavy fuel oil" in p or "bunker" in p or "furnace oil" in p:
        return SPECIFIC_GRAVITY["heavy"]
    return SPECIFIC_GRAVITY["generic"]


def cmd_normalize() -> int:
    """parsed/*.parquet -> normalized/*.parquet — apply FX + UOM conversions."""
    if not FX_PATH.exists():
        print(f"[FAIL] FX cache missing. Run: python {Path(__file__).name} fx")
        return 1

    NORM_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PARSED_DIR.glob("chunk_*.parquet"))
    if not files:
        print(f"[FAIL] No parsed chunks in {PARSED_DIR}. Run parse first.")
        return 1

    fx = pd.read_parquet(FX_PATH)
    fx["DATE"] = pd.to_datetime(fx["DATE"])
    print(f"[NORM] {len(files)} chunks -> {NORM_DIR}  (FX rows: {len(fx):,})")

    for i, f in enumerate(files, 1):
        out_path = NORM_DIR / f.name
        if out_path.exists():
            print(f"  [{i:>3}/{len(files)}] SKIP   {f.name}")
            continue

        t0 = time.time()
        df = pd.read_parquet(f)
        # Normalize column casing: parsed files written before the uppercase fix
        # have Title-case parser columns (Product/Grade/...); rename to UPPER.
        rename_map = {c: c.upper() for c in df.columns if c != c.upper() and c.upper() not in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"])
        df["_DATE"] = df["ASSESSDATE"].dt.normalize()

        # Phase 1: Currency -> USD
        merged = df.merge(fx, left_on="_DATE", right_on="DATE", how="left")
        v = merged["VALUE"].astype(float)
        cur = merged["CURRENCY"].fillna("USD")

        usd_value = np.select(
            [
                cur == "USC",                    # US cents
                cur == "CAC",                    # CAD cents
                cur == "CAD",                    # CAD dollars
                cur == "EUR",
                cur == "USD",
            ],
            [
                v / 100.0,
                (v / 100.0) * merged["CAD_TO_USD"],
                v * merged["CAD_TO_USD"],
                v * merged["EUR_TO_USD"],
                v,
            ],
            default=v,
        )

        # Phase 2/3: UOM -> USD/GAL
        bbl_per_mt = merged["PRODUCT"].apply(_bbl_per_mt)
        uom = merged["UOM"].fillna("")

        norm_value = np.select(
            [uom == "GAL", uom == "LTR", uom == "BBL", uom == "MT"],
            [
                usd_value,
                usd_value * 3.78541,
                usd_value / 42.0,
                (usd_value / bbl_per_mt) / 42.0,
            ],
            default=usd_value,
        )

        df["USD_VALUE"]                  = usd_value
        df["NORMALIZED_VALUE_USD_GAL"]   = norm_value
        df["NORMALIZED_UOM"]             = "GAL"
        df["NORMALIZED_CURRENCY"]        = "USD"
        df = df.drop(columns=["_DATE"])

        df.to_parquet(out_path, compression="snappy", index=False)
        print(f"  [{i:>3}/{len(files)}] OK     {f.name}  {len(df):>10,} rows  {time.time()-t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# STAGE 3 — 256-day rolling Z-score per symbol
# ---------------------------------------------------------------------------

def cmd_zscore() -> int:
    """All normalized/*.parquet -> zscore/pricedata_ml_ready.parquet.

    Mirrors commodity/utility/revin_sql.py:
      1. Median-dedupe per (SYMBOL, ASSESSDATE)
      2. Dense per-symbol daily grid (start..end of that symbol's life)
      3. Forward-fill across gaps
      4. 256-day rolling mean & std per SYMBOL
      5. Z = clip(-3, 3, (val - mean) / (std + 1e-8))
    """
    ZSCORE_DIR.mkdir(parents=True, exist_ok=True)
    if ZSCORE_OUT.exists():
        print(f"[ZSCORE] Output exists: {ZSCORE_OUT}  (delete it to recompute)")
        return 0

    files = sorted(NORM_DIR.glob("chunk_*.parquet"))
    if not files:
        print(f"[FAIL] No normalized chunks in {NORM_DIR}. Run normalize first.")
        return 1

    print(f"[ZSCORE] Loading {len(files)} normalized chunks...")
    t0 = time.time()
    keep = [
        "SYMBOL", "ASSESSDATE", "NORMALIZED_VALUE_USD_GAL",
        "PRODUCT", "GRADE", "GEOGRAPHY", "DELIVERY", "TIMING",
    ]
    df = pd.concat([pd.read_parquet(f, columns=keep) for f in files], ignore_index=True)
    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"]).dt.normalize()
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # 1. Median-dedupe per (SYMBOL, ASSESSDATE)
    print("[ZSCORE] Median-dedupe per (SYMBOL, ASSESSDATE)...")
    t0 = time.time()
    agg = (df.groupby(["SYMBOL", "ASSESSDATE"], sort=False)
             .agg(NORMALIZED_VALUE_USD_GAL=("NORMALIZED_VALUE_USD_GAL", "median"),
                  PRODUCT=("PRODUCT", "first"),
                  GRADE=("GRADE", "first"),
                  GEOGRAPHY=("GEOGRAPHY", "first"),
                  DELIVERY=("DELIVERY", "first"),
                  TIMING=("TIMING", "first"))
             .reset_index())
    print(f"  Deduped to {len(agg):,} (SYMBOL, DATE) rows in {time.time()-t0:.1f}s")

    # 2-3. Dense per-symbol grid + forward-fill
    print("[ZSCORE] Dense per-symbol daily grid + ffill...")
    t0 = time.time()
    dense_pieces = []
    bounds = agg.groupby("SYMBOL")["ASSESSDATE"].agg(["min", "max"])
    for sym, sub in agg.groupby("SYMBOL", sort=False):
        d0, d1 = bounds.at[sym, "min"], bounds.at[sym, "max"]
        grid = pd.date_range(d0, d1, freq="D")
        out = (sub.set_index("ASSESSDATE")
                  .reindex(grid))
        out["SYMBOL"] = sym
        out["NORMALIZED_VALUE_USD_GAL"] = out["NORMALIZED_VALUE_USD_GAL"].ffill()
        for c in ("PRODUCT", "GRADE", "GEOGRAPHY", "DELIVERY", "TIMING"):
            out[c] = out[c].ffill().bfill()
        out = out.reset_index().rename(columns={"index": "ASSESSDATE"})
        dense_pieces.append(out)
    dense = pd.concat(dense_pieces, ignore_index=True)
    print(f"  Dense grid: {len(dense):,} rows in {time.time()-t0:.1f}s")

    # 4-5. 256-day rolling stats & Z
    print(f"[ZSCORE] Rolling {ROLLING_WINDOW}-day mean/std per SYMBOL...")
    t0 = time.time()
    dense = dense.sort_values(["SYMBOL", "ASSESSDATE"]).reset_index(drop=True)
    grouped = dense.groupby("SYMBOL", sort=False)["NORMALIZED_VALUE_USD_GAL"]
    dense["ROLLING_MEAN"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    dense["ROLLING_STD"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).std()
    )
    z = (dense["NORMALIZED_VALUE_USD_GAL"] - dense["ROLLING_MEAN"]) / (dense["ROLLING_STD"] + 1e-8)
    dense["Z_SCORE"] = z.clip(-3.0, 3.0).fillna(0.0)
    print(f"  Rolling stats in {time.time()-t0:.1f}s")

    # Save
    print(f"[ZSCORE] Writing {ZSCORE_OUT} ...")
    out_cols = ["SYMBOL", "ASSESSDATE", "NORMALIZED_VALUE_USD_GAL",
                "ROLLING_MEAN", "ROLLING_STD", "Z_SCORE",
                "PRODUCT", "GRADE", "GEOGRAPHY", "DELIVERY", "TIMING"]
    dense[out_cols].to_parquet(ZSCORE_OUT, compression="snappy", index=False)
    print(f"[OK] Wrote {len(dense):,} rows ({ZSCORE_OUT.stat().st_size/1e6:.1f} MB)")
    return 0


# ---------------------------------------------------------------------------
# STAGE 4 — all
# ---------------------------------------------------------------------------

def cmd_all() -> int:
    for step in (cmd_parse, cmd_fx, cmd_normalize, cmd_zscore):
        rc = step()
        if rc != 0:
            return rc
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("parse",     help="raw/*.parquet -> parsed/*.parquet (DESCRIPTION regex)")
    fx_p = sub.add_parser("fx", help="Download FRED CAD/EUR rates")
    fx_p.add_argument("--force", action="store_true", help="Re-download even if cached")
    sub.add_parser("normalize", help="parsed/*.parquet -> normalized/*.parquet (FX + UOM)")
    sub.add_parser("zscore",    help="normalized/*.parquet -> zscore/pricedata_ml_ready.parquet")
    sub.add_parser("all",       help="Run parse + fx + normalize + zscore")

    args = p.parse_args()
    if args.cmd == "parse":     return cmd_parse()
    if args.cmd == "fx":        return cmd_fx(force=args.force)
    if args.cmd == "normalize": return cmd_normalize()
    if args.cmd == "zscore":    return cmd_zscore()
    if args.cmd == "all":       return cmd_all()
    return 2


if __name__ == "__main__":
    sys.exit(main())
