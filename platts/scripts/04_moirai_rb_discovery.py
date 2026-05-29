"""MOIRAI discovery with RB as the fixed anchor.

Reads:
    platts/outputs/pricedata/zscore/pricedata_ml_ready.parquet   (PLATTS Z-scores)
    platts/outputs/rb_returns.parquet                            (RB log returns)

For every batch of PLATTS candidates we splice in RB's Z-scored log return as
an additional variate and run a MOIRAI multivariate forecast. We capture the
encoder's last-layer cross-attention weights and read the attention FROM RB's
query tokens TO each other variate's key/value tokens. That number is RB's
"how much do I rely on you" score for each PLATTS symbol.

Output: platts/outputs/moirai_rb_ranking.csv  (ranked by attention into RB)

Usage:
    python platts/scripts/04_moirai_rb_discovery.py
    python platts/scripts/04_moirai_rb_discovery.py --batch-size 12 --top-k 50
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

ZSCORE_PATH    = PROJECT_ROOT / "platts" / "outputs" / "pricedata" / "zscore" / "pricedata_ml_ready.parquet"
RB_PATH        = PROJECT_ROOT / "platts" / "outputs" / "rb_returns.parquet"
OUT_PATH       = PROJECT_ROOT / "platts" / "outputs" / "moirai_rb_ranking.csv"

# Defaults mirror commodity/ml/inference.py
DEFAULT_MODEL_ID         = "Salesforce/moirai-1.1-R-small"
DEFAULT_CONTEXT_LENGTH   = 200
DEFAULT_PRED_LENGTH      = 10
DEFAULT_BATCH_SIZE       = 19          # 19 PLATTS + 1 RB per batch = 20 total
DEFAULT_LIQUIDITY        = 0.80
DEFAULT_TOP_K            = 30
DEFAULT_ROLL_WINDOW      = 256         # match PLATTS Z-score window
DEFAULT_MAX_CANDIDATES   = 500         # cap symbols sent to MOIRAI (variance-rank)

# RB anchor name as a "virtual symbol" we insert into the wide matrix
RB_SYMBOL = "__RB_ZSCORE__"

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(prefer: str = "auto") -> str:
    """Pick device for MOIRAI inference.

    NOTE: Apple MPS does NOT support float64, which MOIRAI uses internally for
    its normalization statistics. MPS will hard-fail every batch. So `auto`
    deliberately skips MPS and falls back to CPU on Apple Silicon. Users who
    really want to try MPS can pass --device mps explicitly.
    """
    import torch
    if prefer == "cpu":
        return "cpu"
    if prefer == "mps":
        return "mps"  # user explicitly asked; let it fail loud if needed
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    # auto on Apple Silicon -> CPU (MPS is float32-only, MOIRAI needs float64)
    return "cpu"


# ---------------------------------------------------------------------------
# Attention capture hook (mirrors commodity/ml/inference.py exactly)
# ---------------------------------------------------------------------------
_attn_store: dict = {}


def _capture_attention_hook(module, args, kwargs, output):
    import torch
    from einops import rearrange, repeat
    raw_q, raw_k = args[0], args[1]
    query_var_id = kwargs.get("query_var_id")
    kv_var_id    = kwargs.get("kv_var_id")
    with torch.no_grad():
        q = module.q_proj(raw_q)
        k = module.k_proj(raw_k)
        q = module.q_norm(rearrange(
            q, "... q_len (group hpg dim) -> ... group hpg q_len dim",
            group=module.num_groups, hpg=module.heads_per_group))
        k = module.k_norm(repeat(
            k, "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=module.num_groups, hpg=module.heads_per_group))
        scores = (q @ k.transpose(-2, -1)) * module.softmax_scale
        weights = torch.softmax(scores, dim=-1)
        avg_weights = weights.mean(dim=(-4, -3))
    _attn_store["weights"]      = avg_weights.detach().cpu()
    _attn_store["query_var_id"] = query_var_id.detach().cpu() if query_var_id is not None else None
    _attn_store["kv_var_id"]    = kv_var_id.detach().cpu()    if kv_var_id    is not None else None


# ---------------------------------------------------------------------------
# Data loading + alignment
# ---------------------------------------------------------------------------

def load_platts_wide(
    liquidity: float,
    max_candidates: int,
    product_filter: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pivot_df indexed by date, symbol_meta keyed by SYMBOL).

    Narrows symbols in three steps:
      1. (optional) PRODUCT substring filter
      2. Liquidity >= threshold
      3. Variance-rank: keep top max_candidates most volatile
    """
    print(f"[LOAD] {ZSCORE_PATH.name} ...")
    df = pd.read_parquet(ZSCORE_PATH, columns=[
        "SYMBOL", "ASSESSDATE", "Z_SCORE", "PRODUCT", "GRADE", "GEOGRAPHY"
    ])
    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"]).dt.normalize()

    meta = (df.drop_duplicates("SYMBOL", keep="first")
              .set_index("SYMBOL")[["PRODUCT", "GRADE", "GEOGRAPHY"]])
    print(f"       {len(df):,} rows, {df['SYMBOL'].nunique():,} unique symbols")

    # Step 1: optional PRODUCT substring filter (case-insensitive)
    if product_filter:
        pf_lower = [p.lower() for p in product_filter]
        meta_prod_lower = meta["PRODUCT"].fillna("").str.lower()
        keep_by_product = meta_prod_lower.apply(
            lambda x: any(p in x for p in pf_lower)
        )
        keep_syms = meta.index[keep_by_product].tolist()
        df = df[df["SYMBOL"].isin(keep_syms)]
        print(f"       Product filter {product_filter}: kept {len(keep_syms):,} symbols")

    # Pivot (NOTE: do NOT asfreq -- it triggers freq=D enforcement that
    #         breaks downstream union with RB. PandasDataset takes freq separately.)
    pivot = df.pivot(index="ASSESSDATE", columns="SYMBOL", values="Z_SCORE")
    pivot = pivot.sort_index()
    # Build a continuous daily grid manually (no freq metadata on the index)
    full = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
    pivot = pivot.reindex(full)
    pivot.index.name = "ASSESSDATE"

    # Step 2: Liquidity filter
    valid = pivot.notna() & (pivot != 0)
    keep_liq = valid.mean()[valid.mean() >= liquidity].index.tolist()
    pivot = pivot[keep_liq]
    pivot = pivot.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    print(f"       Liquidity >= {liquidity:.0%}: {len(keep_liq):,} symbols")

    # Step 3: Variance-rank top-N
    if max_candidates and len(keep_liq) > max_candidates:
        top_var = pivot.var().sort_values(ascending=False).head(max_candidates).index.tolist()
        pivot = pivot[top_var]
        print(f"       Top {max_candidates} by variance: kept {len(top_var):,} symbols")

    return pivot, meta.loc[meta.index.intersection(pivot.columns)]


def load_rb_zscore(window: int) -> pd.Series:
    """Load RB log-return and apply trailing window-day rolling Z-score."""
    print(f"[LOAD] {RB_PATH.name} ...")
    df = pd.read_parquet(RB_PATH)
    df.index = pd.to_datetime(df.index).normalize()
    r = df["log_return"]
    mu = r.rolling(window, min_periods=1).mean()
    sd = r.rolling(window, min_periods=1).std()
    z = ((r - mu) / (sd + 1e-8)).clip(-3.0, 3.0).fillna(0.0)
    print(f"       RB rolling Z: {len(z):,} obs, "
          f"{z.index.min().date()} -> {z.index.max().date()}")
    return z


def align(pivot: pd.DataFrame, rb_z: pd.Series) -> pd.DataFrame:
    """Outer-join PLATTS pivot with RB on a continuous daily grid; ffill, fill 0."""
    rb_z = rb_z[~rb_z.index.duplicated(keep="last")]
    rb_z.name = RB_SYMBOL

    # Daily grid spanning both series
    start = min(pivot.index.min(), rb_z.index.min())
    end   = max(pivot.index.max(), rb_z.index.max())
    grid  = pd.date_range(start, end, freq="D")

    out = pivot.reindex(grid)
    out[RB_SYMBOL] = rb_z.reindex(grid)
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    out = out.dropna(axis=1, how="all")
    out.index.name = "ASSESSDATE"
    # PandasDataset takes freq via its own kwarg, no need to set on the index.
    print(f"[ALIGN] Daily grid: {len(out):,} days x {out.shape[1]:,} variates "
          f"(includes RB anchor)")
    return out


# ---------------------------------------------------------------------------
# MOIRAI scan with RB anchor
# ---------------------------------------------------------------------------

def run_moirai_scan(
    aligned: pd.DataFrame,
    model_id: str,
    context_length: int,
    pred_length: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Return {symbol: attention_score_from_RB} aggregated across batches."""
    import torch
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    print(f"[MOIRAI] Loading {model_id} on {device} ...")
    module = MoiraiModule.from_pretrained(model_id)

    other_symbols = [c for c in aligned.columns if c != RB_SYMBOL]
    influence: dict[str, float] = {}

    n_batches = -(-len(other_symbols) // batch_size)
    for bi in range(n_batches):
        platts_chunk = other_symbols[bi * batch_size : (bi + 1) * batch_size]
        if not platts_chunk:
            continue

        # Build a batch: RB + N PLATTS symbols
        chunk_symbols = [RB_SYMBOL, *platts_chunk]
        num_in_chunk = len(chunk_symbols)
        print(f"  Batch {bi+1}/{n_batches}: RB + {len(platts_chunk)} PLATTS ...", end=" ")

        chunk_df = aligned[chunk_symbols].copy()
        ds = PandasDataset(chunk_df, target=chunk_symbols, freq="D")

        model = MoiraiForecast(
            module=module,
            prediction_length=pred_length,
            context_length=context_length,
            patch_size="auto",
            num_samples=1,
            target_dim=num_in_chunk,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        last_attn = model.module.encoder.layers[-1].self_attn
        hook = last_attn.register_forward_hook(_capture_attention_hook, with_kwargs=True)
        predictor = model.create_predictor(batch_size=1, device=device)

        try:
            list(predictor.predict(ds))
            weights = _attn_store["weights"][0]      # (q_len, kv_len)
            q_var   = _attn_store["query_var_id"][0] # (q_len,)
            kv_var  = _attn_store["kv_var_id"][0]    # (kv_len,)

            # RB is variate index 0 in this chunk (we put it first)
            rb_q_mask = (q_var == 0)
            if rb_q_mask.any():
                for j in range(1, num_in_chunk):  # skip self (j=0)
                    kv_mask = (kv_var == j)
                    if kv_mask.any():
                        attn = weights[rb_q_mask][:, kv_mask].mean().item()
                        influence[chunk_symbols[j]] = attn
            print(f"OK (captured {len(platts_chunk)} scores)")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")
        finally:
            hook.remove()
            del model, predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return influence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id",       default=DEFAULT_MODEL_ID)
    p.add_argument("--context-length", type=int,   default=DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--pred-length",    type=int,   default=DEFAULT_PRED_LENGTH)
    p.add_argument("--batch-size",     type=int,   default=DEFAULT_BATCH_SIZE,
                   help="PLATTS symbols per batch (+ 1 for RB anchor)")
    p.add_argument("--liquidity",      type=float, default=DEFAULT_LIQUIDITY)
    p.add_argument("--top-k",          type=int,   default=DEFAULT_TOP_K)
    p.add_argument("--roll-window",    type=int,   default=DEFAULT_ROLL_WINDOW,
                   help="Window for RB log-return rolling Z (default 256, matches PLATTS)")
    p.add_argument("--max-candidates", type=int,   default=DEFAULT_MAX_CANDIDATES,
                   help="Cap on # of PLATTS symbols sent to MOIRAI (top-N by variance). "
                        "0 disables the cap.")
    p.add_argument("--product-filter", default=None,
                   help="Comma-separated PRODUCT substrings (case-insensitive). "
                        "e.g. 'gasoline,naphtha,rbob,wti,brent,ulsd,heating oil'")
    p.add_argument("--device",         default="auto", choices=["auto","cuda","mps","cpu"])
    args = p.parse_args()

    product_filter = (
        [s.strip() for s in args.product_filter.split(",") if s.strip()]
        if args.product_filter else None
    )

    device = select_device(args.device)
    print(f"[CONFIG] device={device}  model={args.model_id}  "
          f"ctx={args.context_length}  pred={args.pred_length}  "
          f"batch={args.batch_size}  liq>={args.liquidity:.0%}\n")

    pivot, meta = load_platts_wide(
        liquidity=args.liquidity,
        max_candidates=args.max_candidates,
        product_filter=product_filter,
    )
    rb_z = load_rb_zscore(window=args.roll_window)
    aligned = align(pivot, rb_z)

    influence = run_moirai_scan(
        aligned,
        model_id=args.model_id,
        context_length=args.context_length,
        pred_length=args.pred_length,
        batch_size=args.batch_size,
        device=device,
    )

    # Build ranking
    rank_df = (pd.Series(influence, name="ATTENTION_FROM_RB")
                 .sort_values(ascending=False)
                 .reset_index().rename(columns={"index": "SYMBOL"}))
    rank_df["RANK"] = np.arange(1, len(rank_df) + 1)
    rank_df = rank_df.join(meta, on="SYMBOL")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Wrote ranking for {len(rank_df):,} symbols -> {OUT_PATH}")

    # Show top-K
    print(f"\n{'='*78}\n  TOP {args.top_k} PLATTS DRIVERS OF RB (by attention from RB)\n{'='*78}")
    print(f"  {'RANK':>4}  {'SYMBOL':<10}  {'SCORE':>8}  {'PRODUCT':<28}  {'GRADE':<12}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*28}  {'-'*12}")
    for _, row in rank_df.head(args.top_k).iterrows():
        prod  = (str(row.get("PRODUCT") or "")[:28])
        grade = (str(row.get("GRADE")   or "")[:12])
        bar = "#" * int(row["ATTENTION_FROM_RB"] /
                        rank_df["ATTENTION_FROM_RB"].iloc[0] * 20)
        print(f"  {int(row['RANK']):>4}  {row['SYMBOL']:<10}  "
              f"{row['ATTENTION_FROM_RB']:>8.4f}  {prod:<28}  {grade:<12}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
