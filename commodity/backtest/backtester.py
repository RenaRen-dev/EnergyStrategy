"""
Walk-Forward Optimization Backtester for Project 1.

ARCHITECTURE:
- 6-Year WFO (2016-2021), 1-year lookback (2015).
- Monthly Rebalance: Triggers MOIRAI asset selection over trailing 252 days.
- Daily Execution: Triggers Chronos inference using strictly T-1 history.
- Allocates capital dynamically based on MOIRAI Attention scores.
- Automatic caching of WFO outputs to avoid redundant heavy ML runs.
"""

import os
import sys
import warnings
import random
import datetime
import json
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. MONKEYPATCH PANDAS FOR QUANTSTATS COMPATIBILITY
# ==============================================================================
# Under pandas < 2.2.0 (like 2.1.4), the 'ME' frequency alias triggers a ValueError.
# Newer quantstats releases use 'ME' by default, which causes crashes.
# We deeply monkeypatch the pandas to_offset utility in all imported modules to map 'ME' -> 'M' dynamically.
def patch_pandas_to_offset():
    try:
        import pandas._libs.tslibs.offsets as _offsets
        original_to_offset = _offsets.to_offset
        
        def patched_to_offset(freq, *args, **kwargs):
            if isinstance(freq, str):
                if freq == 'ME':
                    freq = 'M'
                elif freq == 'QE':
                    freq = 'Q'
                elif freq == 'YE':
                    freq = 'Y'
            return original_to_offset(freq, *args, **kwargs)
            
        # Overwrite 'to_offset' in every loaded pandas and quantstats module to intercept all calls
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith('pandas') or mod_name.startswith('quantstats'):
                if mod is not None:
                    if hasattr(mod, 'to_offset'):
                        try:
                            setattr(mod, 'to_offset', patched_to_offset)
                        except Exception:
                            pass
                    if hasattr(mod, 'offsets') and hasattr(mod.offsets, 'to_offset'):
                        try:
                            setattr(mod.offsets, 'to_offset', patched_to_offset)
                        except Exception:
                            pass
                            
        _offsets.to_offset = patched_to_offset
        print("Pandas 'ME' compatibility monkeypatched successfully in all modules.")
    except Exception as e:
        print(f"Warning: Could not patch pandas ME compatibility: {e}")

# Apply monkeypatch immediately after imports
patch_pandas_to_offset()

# Add the directory to path to ensure clean import of ai_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new risk engine
try:
    from risk_engine import apply_risk_defense
except ImportError:
    print("Warning: risk_engine.py not found. Risk defenses will be disabled.")
    def apply_risk_defense(ts, tk, ht, cn): return ts, 0.0

# Global cache for Chronos base pipeline to avoid reloading the weights daily
CHRONOS_PIPELINE = None
TRANSACTION_COST_BPS = 10.0



# ==============================================================================
# 1. REAL ML INTEGRATIONS WITH ROBUST ERROR HANDLING
# ==============================================================================

def get_moirai_target_basket(history_df):
    """
    Calls the real MOIRAI pipeline inside ai_inference.py using the provided history_df.
    Returns dynamically vetted 'Follower' refiners and their raw attention scores.
    """
    try:
        from ai_inference import run_moirai_discovery
        
        # Strictly run the actual model discovery
        stock_analysis, _, _ = run_moirai_discovery(history_df)
        
        # Vet only tradeable Followers
        selected = [k for k, v in stock_analysis.items() if v.get('tradeable', False)]
        scores = {k: stock_analysis[k].get('stock_to_crack', 0.01) for k in selected}
        
        return selected, scores
    except Exception as e:
        import traceback
        print(f"  [MOIRAI Exception] Failed discovery at {history_df.index[-1].date()}: {e}")
        traceback.print_exc()
        print("  Skipping rebalance for this month due to MOIRAI failure.")
        return [], {}


def vet_moirai_candidates_with_ablation(history_df, candidate_tickers):
    """
    Stage 3 Vetting: Simulates an ablation test over recent history to validate MOIRAI candidates.
    A candidate is approved ONLY if:
      - Step A: crack321 contribution percentage > 0.
      - Step B & C: Multivariate Hit Rate > Univariate Baseline Hit Rate.
    """
    vetted_tickers = []
    
    try:
        from ai_inference import run_chronos_inference
        import numpy as np
        
        global CHRONOS_PIPELINE
        if CHRONOS_PIPELINE is None:
            from chronos import BaseChronosPipeline
            import torch
            CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", 
                device_map="cuda" if torch.cuda.is_available() else "cpu", 
                dtype=torch.float32
            )
            
        # Use the last 21 trading days of history_df for the ablation test to save time
        if len(history_df) > 80:
            ablation_test_dates = history_df.index[-21:]
        else:
            return candidate_tickers # Not enough history to test
            
        for ticker in candidate_tickers:
            stock_col = f"{ticker}_Hedged_Return"
            if stock_col not in history_df.columns:
                print(f"  [Ablation Warning] {stock_col} missing from history. Skipping {ticker}.")
                continue
                
            baseline_correct = 0
            mv_correct = 0
            baseline_maes = []
            mv_maes = []
            
            print(f"  [{ticker}] Running ablation over last 21 days...")
            for current_date in ablation_test_dates:
                print(f"\r    [Ablation] Inferencing {ticker} on {current_date.date()}...", end="", flush=True)
                hist = history_df[history_df.index < current_date]
                if len(hist) < 60:
                    continue
                    
                actual_ret = history_df.loc[current_date, stock_col]
                actual_dir = 1 if actual_ret > 0 else 0
                
                # Baseline (Univariate)
                pred_base = run_chronos_inference(CHRONOS_PIPELINE, hist, stock_col, [])
                base_mae = abs(pred_base['q50'] - actual_ret)
                baseline_maes.append(base_mae)
                if (pred_base['p_up'] > 0.5) == actual_dir:
                    baseline_correct += 1
                    
                # Multivariate (with Crack Spread)
                pred_mv = run_chronos_inference(CHRONOS_PIPELINE, hist, stock_col, ["Crack_Z_Score"])
                mv_mae = abs(pred_mv['q50'] - actual_ret)
                mv_maes.append(mv_mae)
                if (pred_mv['p_up'] > 0.5) == actual_dir:
                    mv_correct += 1
                    
            print()
            if not baseline_maes:
                print(f"  [Ablation Warning] Not enough valid test days for {ticker}.")
                continue
                
            avg_base_mae = np.mean(baseline_maes)
            avg_mv_mae = np.mean(mv_maes)
            mae_improvement = avg_base_mae - avg_mv_mae
            
            base_hit = (baseline_correct / len(baseline_maes)) * 100
            mv_hit = (mv_correct / len(mv_maes)) * 100
            hit_improvement = mv_hit - base_hit
            
            pair_pct = 0.0
            if mae_improvement > 0:
                pair_pct = (mae_improvement / avg_base_mae) * 100
                pair_pct = min(pair_pct, 100.0)
                
            print(f"  -> [{ticker} Vetting] Crack Contrib: {pair_pct:.1f}% | Hit Rate Imp: {hit_improvement:+.1f}%")
            
            if pair_pct > 0 and hit_improvement > 0:
                vetted_tickers.append(ticker)
                print(f"     => {ticker} APPROVED.")
            else:
                print(f"     => {ticker} REJECTED.")
            
    except Exception as e:
        import traceback
        print(f"  [Ablation Exception] Real ablation failed: {e}")
        traceback.print_exc()
        
    return vetted_tickers


def get_chronos_prediction(history_df, ticker):
    """
    Calls the real Chronos-2 pipeline using history_df strictly up to T-1.
    If anything fails, gracefully falls back to 0.5 (flat position) to preserve execution.
    """
    global CHRONOS_PIPELINE
    try:
        if CHRONOS_PIPELINE is None:
            print("  Initializing Chronos-2 base pipeline (first run)...")
            from chronos import BaseChronosPipeline
            import torch
            CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", 
                device_map="cuda" if torch.cuda.is_available() else "cpu", 
                dtype=torch.float32
            )
            
        from ai_inference import run_chronos_inference
        
        target_col = f"{ticker}_Hedged_Return"
        covariate_cols = ["Crack_Z_Score"]
        
        # Strict validation of data requirements
        if target_col not in history_df.columns or "Crack_Z_Score" not in history_df.columns:
            return 0.5
            
        pred = run_chronos_inference(
            CHRONOS_PIPELINE, 
            history_df, 
            target_col=target_col, 
            covariate_cols=covariate_cols
        )
        
        p_up = pred.get('p_up', 0.5)
        if pd.isna(p_up) or p_up is None:
            return 0.5
        return p_up
    except Exception as e:
        import traceback
        print(f"  [Chronos Error on {ticker}]: {e}")
        traceback.print_exc()
        # Fallback to neutral 0.5 on any runtime error (flat position size)
        return 0.5


# ==============================================================================
# 2. WFO BACKTEST ENGINE
# ==============================================================================

class WFOEngine:
    def __init__(self, data_path, notional=100):
        self.data_path = data_path
        self.notional = notional
        self.master_df = None
        self.current_positions = {}
        self.total_transaction_costs = 0.0
        self.trade_logs = []
        self.ticker_pnl_log = []
        self.load_data()
        
    def load_data(self):
        print(f"Loading master dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        # Allow full production run from 2015 warmup to 2021 end
        self.master_df = df[(df.index >= '2015-01-01') & (df.index <= '2021-12-31')]
        
    def run(self):
        print("=" * 80)
        print(" PRODUCTION: WFO ENGINE (2016-2021)")
        print("=" * 80)
        
        out_dir = os.path.dirname(self.data_path)
        cache_returns_path = os.path.join(out_dir, "wfo_returns_cache.csv")
        cache_alloc_path = os.path.join(out_dir, "wfo_allocation_cache.json")
        
        # Check if cache exists to avoid heavy re-running of models
        # TEMPORARILY DISABLED during testing so the ablation logic always runs!
        # if os.path.exists(cache_returns_path) and os.path.exists(cache_alloc_path):
        #     print("\n>>> Detected cached backtest outcome files.")
        #     print(f" -> {cache_returns_path}")
        #     print(f" -> {cache_alloc_path}")
        #     print(">>> Loading backtest results from cache to skip heavy ML loops...")
        #     
        #     try:
        #         daily_returns = pd.read_csv(cache_returns_path, index_col=0, parse_dates=True).iloc[:, 0]
        #         daily_returns.name = "Strategy"
        #         
        #         with open(cache_alloc_path, 'r') as f:
        #             serializable_log = json.load(f)
        #         
        #         allocation_log = []
        #         for log in serializable_log:
        #             allocation_log.append({
        #                 'Date': pd.Timestamp(log['Date']),
        #                 'Weights': log['Weights']
        #             })
        #         
        #         print("Cache loaded successfully! Regenerating reports...")
        #         self.generate_tearsheet(daily_returns, allocation_log)
        #         return
        #     except Exception as e:
        #         print(f"Error loading cache: {e}. Re-running full backtest...")
        
        # warmup lookback (strictly calendar year 2015)
        start_date = self.master_df.index[0] 
        warmup_end = pd.Timestamp('2015-12-31')
        if start_date > warmup_end:
            warmup_end = start_date + pd.DateOffset(years=1)
            
        print(f"Warm-up period ends: {warmup_end.date()}")
        
        # Monthly rebalance dates
        valid_df = self.master_df[self.master_df.index > warmup_end]
        month_ends = valid_df.resample('M').last().index
        
        daily_pnl = pd.Series(0.0, index=valid_df.index)
        allocation_log = []
        
        print("\nStarting Monthly WFO Loop...\n")
        
        for i in range(len(month_ends)):
            current_month_end = month_ends[i]
            
            # Identify trading days for the upcoming month
            if i == len(month_ends) - 1:
                test_days = self.master_df[self.master_df.index > current_month_end].index
            else:
                next_month_end = month_ends[i+1]
                test_days = self.master_df[(self.master_df.index > current_month_end) & (self.master_df.index <= next_month_end)].index
                
            if len(test_days) == 0:
                continue
                
            # -- PHASE 1: MOIRAI ASSET SELECTION (Discovery) --
            # Strictly use trailing 252 trading days up to current_month_end
            history_moirai = self.master_df[self.master_df.index <= current_month_end]
            if len(history_moirai) > 252:
                history_moirai = history_moirai.iloc[-252:]
                
            selected_tickers, attention_scores = get_moirai_target_basket(history_moirai)
            
            if not selected_tickers:
                print(f"[{current_month_end.date()}] Skipping rebalance or no tradeable assets returned.")
                continue
                
            # -- PHASE 2: ABLATION VETTING --
            vetted_tickers = vet_moirai_candidates_with_ablation(history_moirai, selected_tickers)
            
            if not vetted_tickers:
                print(f"[{current_month_end.date()}] All MOIRAI candidates failed ablation vetting. Skipping month.")
                continue
                
            # Filter attention_scores to keep only vetted tickers
            vetted_attention_scores = {t: attention_scores[t] for t in vetted_tickers}
                
            # -- PHASE 3: DYNAMIC WEIGHTING --
            # Compute dynamic weights based on MOIRAI raw attention scores for surviving tickers
            total_score = sum(vetted_attention_scores.values())
            weights = {t: score/total_score for t, score in vetted_attention_scores.items()}
            
            w_str = ", ".join([f"{t}: {w*100:.1f}%" for t, w in weights.items()])
            print(f"[{current_month_end.date()}] Rebalance Vetted Basket -> {w_str}")
            
            allocation_log.append({
                'Date': current_month_end,
                'Weights': weights
            })
            
            # -- PHASE 4: CHRONOS DAILY INFERENCE (Execution) --
            monthly_pnl_sum = 0.0
            monthly_friction = 0.0
            for T in test_days:
                # Strictly slice the trailing 365 calendar days up to T-1 to prevent OOM
                lookback_start = T - pd.Timedelta(days=365)
                history_T = self.master_df[(self.master_df.index >= lookback_start) & (self.master_df.index < T)]
                print(f"\r  [Execution] Running Chronos for {T.date()}...", end="", flush=True)
                
                if len(history_T) == 0:
                    continue
                    
                day_pnl = 0.0
                active_allocated_dollars = 0.0
                total_hedge_size = 0.0  # Accumulate tail-risk hedges
                all_active_tickers = set(vetted_tickers).union(self.current_positions.keys())
                
                for ticker in all_active_tickers:
                    target_size = 0.0
                    p_up = 0.5  # Default neutral
                    
                    if ticker in vetted_tickers:
                        hedged_ret_col = f"{ticker}_Hedged_Return"
                        
                        # --- HARD STOP-LOSS: Black Swan Breaker ---
                        # If the asset has crashed >10% in trailing 5 trading days, bypass AI and force liquidation
                        stop_loss_triggered = False
                        if hedged_ret_col in history_T.columns and len(history_T) >= 6:
                            trailing_5d_ret = (
                                (history_T.iloc[-1][hedged_ret_col] + 1) /
                                (history_T.iloc[-6][hedged_ret_col] + 1) - 1
                            )
                            if trailing_5d_ret < -0.10:
                                target_size = 0.0  # Force liquidation / stop-loss
                                stop_loss_triggered = True
                        
                        if not stop_loss_triggered:
                            p_up = get_chronos_prediction(history_T, ticker)
                            
                            # --- AGGRESSIVE DEADBAND: Only trade if extremely confident ---
                            if 0.40 < p_up < 0.60:
                                target_size = 0.0
                            else:
                                # Scale conviction dynamically for tail probabilities
                                conviction = min(1.0, abs(p_up - 0.5) * 5) # Adjusted scale multiplier
                                ticker_capital = self.notional * weights.get(ticker, 0.0)
                                target_size = (1 if p_up > 0.5 else -1) * conviction * ticker_capital
                    
                    # --- APPLY THREE-LAYER RISK DEFENSE ---
                    target_size, current_hedge = apply_risk_defense(target_size, ticker, history_T, self.notional)
                    total_hedge_size += current_hedge
                    
                    current_size = self.current_positions.get(ticker, 0.0)
                    trade_size = target_size - current_size
                    
                    friction_cost = abs(trade_size) * (TRANSACTION_COST_BPS / 10000.0)
                    monthly_friction += friction_cost
                    self.total_transaction_costs += friction_cost
                    
                    active_allocated_dollars += abs(target_size)
                    
                    # Executed at MOC of T-1 and closed at MOC of Day T
                    hedged_ret_col = f"{ticker}_Hedged_Return"
                    if hedged_ret_col in self.master_df.columns:
                        actual_ret = self.master_df.loc[T, hedged_ret_col]
                        asset_pnl = (current_size * actual_ret) - friction_cost
                        day_pnl += asset_pnl
                        
                        self.ticker_pnl_log.append({
                            'Date': T,
                            'Ticker': ticker,
                            'PnL': asset_pnl
                        })
                        
                        # Log trade outcomes for Confidence Scaling Metric (only non-zero, non-stoploss trades)
                        if ticker in vetted_tickers and target_size != 0.0:
                            self.trade_logs.append({
                                'Date': T,
                                'Ticker': ticker,
                                'p_up': p_up,
                                'Actual_Return': actual_ret
                            })
                        
                    # Update State
                    if target_size == 0.0:
                        self.current_positions.pop(ticker, None)
                    else:
                        self.current_positions[ticker] = target_size
                        
                # Core-Satellite (Beta Sweep) Smoothing & Dynamic Hedging
                active_allocated_pct = active_allocated_dollars / self.notional
                spy_ret = self.master_df.loc[T, "SPY_Return"] if "SPY_Return" in self.master_df.columns else 0.0
                
                # Execute emergency short hedge if triggered by Layer 3
                if total_hedge_size != 0.0:
                    hedge_pnl = total_hedge_size * spy_ret
                    day_pnl += hedge_pnl
                    self.ticker_pnl_log.append({
                        'Date': T,
                        'Ticker': 'SPY_Hedge',
                        'PnL': hedge_pnl
                    })
                    print(f"\n      [RISK ENGINE] Emergency SPY Hedge Executed: {total_hedge_size:.2f} at {T.date()}")
                
                elif active_allocated_pct < 1.0 and "SPY_Return" in self.master_df.columns:
                    # Calculate 21-day annualized Realized Volatility of SPY
                    if len(history_T) >= 21:
                        spy_21d_vol = history_T["SPY_Return"].iloc[-21:].std() * np.sqrt(252)
                        spy_21d_vol = max(spy_21d_vol, 0.05) # Floor at 5% to prevent division by zero/hyper-leverage
                    else:
                        spy_21d_vol = 0.15 # Default baseline vol
                        
                    target_vol = 0.15 # Target 15% annualized volatility
                    vol_scalar = target_vol / spy_21d_vol
                    vol_scalar = min(vol_scalar, 1.0) # Never lever up greater than 1x cash
                    
                    # Dynamically shrink the sweep size when the market is crashing/highly volatile
                    spy_sweep_size = self.notional * (1.0 - active_allocated_pct) * vol_scalar
                    
                    spy_pnl = spy_sweep_size * spy_ret
                    day_pnl += spy_pnl
                    
                    self.ticker_pnl_log.append({
                        'Date': T,
                        'Ticker': 'SPY_Sweep',
                        'PnL': spy_pnl
                    })
                        
                daily_pnl.loc[T] = day_pnl
                monthly_pnl_sum += day_pnl
                
            monthly_ret_pct = (monthly_pnl_sum / self.notional) * 100
            print(f"\n  -> [Outcome] Month {current_month_end.date()}: Net Return = {monthly_ret_pct:+.2f}% | Net PnL = ${monthly_pnl_sum:+.2f} | Mthly Friction: ${monthly_friction:.2f}\n")
                
        # Convert daily return series against total daily notional capital
        daily_returns = daily_pnl / self.notional
        daily_returns.name = "Strategy"
        
        print("\nWFO Backtest Complete.")
        print(f"Total Transaction Costs Incurred: ${self.total_transaction_costs:.2f}")
        
        # Save cache files to disk
        try:
            daily_returns.to_csv(cache_returns_path)
            
            serializable_log = []
            for log in allocation_log:
                serializable_log.append({
                    'Date': log['Date'].strftime('%Y-%m-%d'),
                    'Weights': log['Weights']
                })
            with open(cache_alloc_path, 'w') as f:
                json.dump(serializable_log, f)
                
            print(f"\nSaved backtest outcome cache to:")
            print(f" -> {cache_returns_path}")
            print(f" -> {cache_alloc_path}")
        except Exception as e:
            print(f"Warning: Failed to save backtest cache: {e}")
            
        self.ticker_pnl_df = pd.DataFrame(self.ticker_pnl_log)
        self.generate_tearsheet(daily_returns, allocation_log)

    def generate_tearsheet(self, daily_returns, allocation_log):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        print("\nGenerating Master Institutional Tearsheet...")
        out_dir = os.path.dirname(self.data_path)
        
        # Apply the monkeypatch again to ensure all loaded pandas submodules are updated
        patch_pandas_to_offset()
        
        # ----------------------------------------------------------------------
        # 1. DUAL BENCHMARK DATA SWAP (SP500 & XLE)
        # ----------------------------------------------------------------------
        print("\n" + "="*50)
        print("=== [CRITICAL DEBUG] BENCHMARK ALIGNMENT CHECK ===")
        print(f"Total Columns in master_df: {list(self.master_df.columns)}")
        print("="*50 + "\n")

        # Keep SPY As-Is: Keep extracting benchmark_spy from self.master_df.loc[daily_returns.index, "SPY_Return"]
        if "SPY_Return" in self.master_df.columns:
            benchmark_spy = self.master_df.loc[daily_returns.index, "SPY_Return"]
        else:
            raise KeyError("SPY_Return missing from master_df!")

        # Load External XLE CSV
        xle_path = r"C:\Users\styu0\Energy-Strategy\data\data_backtestproject1\XLE_daily.csv"
        if not os.path.exists(xle_path):
            raise FileNotFoundError(f"XLE CSV file not found at path: {xle_path}")

        xle_df = pd.read_csv(xle_path)
        # Clean column headers
        xle_df.columns = [c.strip().lower() for c in xle_df.columns]
        if 'date' not in xle_df.columns:
            raise KeyError("XLE CSV does not contain a 'date' column.")

        xle_df['date'] = pd.to_datetime(xle_df['date'])
        xle_df.set_index('date', inplace=True)
        xle_df.sort_index(inplace=True)

        if 'close' not in xle_df.columns:
            raise KeyError("XLE CSV does not contain a 'Close' price column.")

        # Compute the daily fractional change since it only contains price data
        xle_returns_full = xle_df['close'].pct_change()

        # Align/filter to backtest execution dates. Fill holiday NaNs with 0.0
        benchmark_xle = xle_returns_full.reindex(daily_returns.index).fillna(0.0)

        # Verify Data Vitality (No More Silent Zeros)
        assert benchmark_xle.std() > 0, "CRITICAL: Loaded XLE benchmark has zero variance! Check data parsing."
        print(f"-> Successfully dynamically injected XLE data. Volatility: {benchmark_xle.std():.6f}")
            
        # ----------------------------------------------------------------------
        # 2. HARDCORE STATISTICS MATRIX (Console)
        # ----------------------------------------------------------------------
        cum_ret = (1 + daily_returns).prod() - 1
        ann_ret = (1 + cum_ret) ** (252 / max(1, len(daily_returns))) - 1
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        cum_wealth = (1 + daily_returns).cumprod()
        peaks = cum_wealth.cummax()
        drawdowns = (cum_wealth - peaks) / peaks
        max_dd = drawdowns.min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
        
        win_rate = (daily_returns > 0).mean() * 100
        
        if self.trade_logs:
            trade_df = pd.DataFrame(self.trade_logs)
            trade_df['Conviction'] = abs(trade_df['p_up'] - 0.5)
            trade_df['Predicted_Dir'] = np.where(trade_df['p_up'] > 0.5, 1, -1)
            trade_df['Actual_Dir'] = np.where(trade_df['Actual_Return'] > 0, 1, -1)
            trade_df['Correct'] = trade_df['Predicted_Dir'] == trade_df['Actual_Dir']
            
            high_conf = trade_df[trade_df['Conviction'] > 0.25]
            low_conf = trade_df[trade_df['Conviction'] <= 0.10]
            
            high_conf_hit = high_conf['Correct'].mean() * 100 if len(high_conf) > 0 else np.nan
            low_conf_hit = low_conf['Correct'].mean() * 100 if len(low_conf) > 0 else np.nan
        else:
            trade_df = pd.DataFrame()
            high_conf_hit, low_conf_hit = np.nan, np.nan
            
        print("=" * 60)
        print(" INSTITUTIONAL WFO PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f" Cumulative Return:      {cum_ret*100:+.2f}%")
        print(f" Annualized Return:      {ann_ret*100:+.2f}%")
        print(f" Annualized Sharpe:      {sharpe:.2f}")
        print(f" Max Drawdown:           {max_dd*100:.2f}%")
        print(f" Calmar Ratio:           {calmar:.2f}")
        print(f" Daily Win Rate:         {win_rate:.1f}%")
        print("-" * 60)
        print(" Annual Sharpe Ratios:")
        for year in daily_returns.index.year.unique():
            yr_ret = daily_returns[daily_returns.index.year == year]
            if len(yr_ret) > 10 and yr_ret.std() > 0:
                yr_sharpe = np.sqrt(252) * yr_ret.mean() / yr_ret.std()
                print(f"   {year}: {yr_sharpe:.2f}")
        print("-" * 60)
        print(f" High Confidence Hit:    {high_conf_hit:.1f}% (N={len(high_conf) if not trade_df.empty else 0})")
        print(f" Low Confidence Hit:     {low_conf_hit:.1f}% (N={len(low_conf) if not trade_df.empty else 0})")
        print("=" * 60)

        # ----------------------------------------------------------------------
        # 3. MASTER INSTITUTIONAL TEARSHEET PLOT
        # ----------------------------------------------------------------------
        fig = plt.figure(figsize=(16, 36))
        gs = GridSpec(6, 1, height_ratios=[2.5, 1.5, 2, 2, 2.5, 2], hspace=0.4)
        
        # Ax1: Cumulative Equity Curve
        ax1 = fig.add_subplot(gs[0])
        cum_strat = (1 + daily_returns).cumprod() * 100
        cum_bench_spy = (1 + benchmark_spy).cumprod() * 100
        cum_bench_xle = (1 + benchmark_xle).cumprod() * 100
        
        ax1.plot(cum_strat.index, cum_strat, label='Strategy Net NAV', color='dodgerblue', lw=2.5)
        ax1.plot(cum_bench_spy.index, cum_bench_spy, label='S&P 500 Benchmark', color='gray', lw=1.5, ls='--')
        ax1.plot(cum_bench_xle.index, cum_bench_xle, label='XLE Energy Benchmark', color='purple', lw=1.5, ls='--')
        
        ax1.fill_between(cum_strat.index, cum_bench_spy, cum_strat, where=(cum_strat > cum_bench_spy), 
                         interpolate=True, color='mediumseagreen', alpha=0.2, label='Outperformance (vs SPY)')
                         
        ax1.set_title('Strategy vs. Dual-Benchmark Cumulative Equity ($100 Init)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Ax2: Underwater Drawdown
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(drawdowns.index, drawdowns * 100, 0, color='crimson', alpha=0.6)
        ax2.axhline(-15, color='black', linestyle='--', lw=2, label='-15% Risk Limit')
        ax2.set_title('Underwater Drawdown (%)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        # Ax3: Strategy Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[2])
        monthly_ret = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        heatmap_data = pd.DataFrame({
            'Year': monthly_ret.index.year,
            'Month': monthly_ret.index.strftime('%b'),
            'Return': monthly_ret.values * 100
        })
        heatmap_matrix = heatmap_data.pivot(index='Year', columns='Month', values='Return')
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_matrix = heatmap_matrix.reindex(columns=[m for m in months_order if m in heatmap_matrix.columns])
        
        sns.heatmap(heatmap_matrix, annot=True, cmap='RdYlGn', center=0.0, fmt=".1f", 
                    ax=ax3, cbar_kws={'label': 'Return (%)'}, linewidths=0.5, annot_kws={"size": 11})
        ax3.set_title('Strategy Monthly Net Returns (%)', fontsize=16, fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylabel('Year', fontsize=12)
        
        # Ax4: MOIRAI Vetted Allocation Heatmap
        ax4 = fig.add_subplot(gs[3])
        if allocation_log:
            alloc_df = pd.DataFrame(index=[log['Date'] for log in allocation_log])
            universe = ['PSX', 'CVI', 'VLO', 'MPC', 'DINO', 'PBF', 'DK']
            for ticker in universe:
                alloc_df[ticker] = 0.0
                
            for log in allocation_log:
                for t, w in log['Weights'].items():
                    alloc_df.loc[log['Date'], t] = w * 100
                    
            alloc_df.index = alloc_df.index.strftime('%Y-%m')
            alloc_matrix = alloc_df.T
            
            # Format blank cells for 0%
            annot_matrix = alloc_matrix.applymap(lambda x: f"{x:.1f}%" if x > 0 else "")
            
            sns.heatmap(alloc_matrix, annot=annot_matrix, fmt="", cmap='Blues', ax=ax4, 
                        cbar_kws={'label': 'Capital Weight (%)'}, linewidths=0.5)
            ax4.set_title('MOIRAI Vetted Allocation Heatmap (%)', fontsize=16, fontweight='bold')
            ax4.set_xlabel('Rebalance Month', fontsize=12)
            ax4.set_ylabel('Ticker', fontsize=12)
            
        # Ax5: Realized Monthly PnL Attribution
        ax5 = fig.add_subplot(gs[4])
        if getattr(self, 'ticker_pnl_df', None) is not None and not self.ticker_pnl_df.empty:
            pnl_df = self.ticker_pnl_df.copy()
            pnl_df['Month'] = pnl_df['Date'].dt.to_period('M')
            monthly_pnl = pnl_df.groupby(['Month', 'Ticker'])['PnL'].sum().unstack(fill_value=0)
            
            monthly_pnl.index = monthly_pnl.index.astype(str)
            monthly_pnl.plot(kind='bar', stacked=True, ax=ax5, colormap='tab10', alpha=0.85)
            ax5.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
            ax5.set_title("Realized Monthly Dollar PnL Attribution", fontsize=16, fontweight='bold')
            ax5.set_ylabel("Net PnL ($)", fontsize=12)
            ax5.set_xlabel("Month", fontsize=12)
            ax5.legend(title='Ticker', loc='upper left', bbox_to_anchor=(1.02, 1))
            ax5.tick_params(axis='x', rotation=45)
            
        # Ax6: Rolling 252-Day Annualized Sharpe Ratio
        ax6 = fig.add_subplot(gs[5])
        
        # 1. Calculate 252-Day Rolling Sharpe
        rolling_window = 252
        roll_mean = daily_returns.rolling(window=rolling_window).mean()
        roll_std = daily_returns.rolling(window=rolling_window).std()
        
        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        rolling_sharpe = (roll_mean / roll_std) * np.sqrt(252)
        
        # 2. Plotting
        ax6.plot(rolling_sharpe.index, rolling_sharpe, color='darkorange', lw=2)
        ax6.axhline(1.0, color='black', linestyle='--', lw=1.5, label='Good Sharpe Threshold (1.0)')
        ax6.axhline(0.0, color='gray', linestyle='-', lw=1)
        
        # Fill color depending on above/below zero
        ax6.fill_between(rolling_sharpe.index, rolling_sharpe, 0, where=(rolling_sharpe > 0), color='forestgreen', alpha=0.3)
        ax6.fill_between(rolling_sharpe.index, rolling_sharpe, 0, where=(rolling_sharpe <= 0), color='crimson', alpha=0.3)
        
        ax6.set_title('Rolling 252-Day Annualized Sharpe Ratio', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio', fontsize=12)
        ax6.set_xlabel('Date', fontsize=12)
        ax6.legend(loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        master_path = os.path.join(out_dir, "master_wfo_tearsheet.png")
        plt.savefig(master_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Master Tearsheet Saved: {master_path}")
        
if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "master_dataset.csv")
    engine = WFOEngine(DATA_PATH, notional=100)
    engine.run()
