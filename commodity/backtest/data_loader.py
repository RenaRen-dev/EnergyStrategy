import os
import glob
import pandas as pd
import numpy as np
import warnings

# Suppress pandas FutureWarning for fillna(method='bfill')
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data_backtestproject1"))
FUTURES_DIR = os.path.join(DATA_DIR, "futures")

def build_continuous_futures(commodity, M=3):
    """
    Builds a continuous series for the M-th nearby contract.
    Loads all CSVs for the commodity, and for each unique trade_date,
    selects the settlement price of the contract that is the M-th to expire.
    """
    files = glob.glob(os.path.join(FUTURES_DIR, f"{commodity}_*.csv"))
    df_list = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['date', 'expiry_date'])
        df_list.append(df)
        
    if not df_list:
        return pd.Series(dtype=float)
        
    all_data = pd.concat(df_list, ignore_index=True)
    all_data = all_data[all_data['date'] <= all_data['expiry_date']]
    
    # Sort by date, then by expiry_date to correctly order promptness
    all_data.sort_values(['date', 'expiry_date'], inplace=True)
    
    grouped = all_data.groupby('date')
    
    continuous_dates = []
    continuous_settle = []
    
    for date, group in grouped:
        if len(group) >= M:
            continuous_dates.append(date)
            continuous_settle.append(group.iloc[M-1]['settlement'])
            
    return pd.Series(continuous_settle, index=continuous_dates, name=f"{commodity}_M{M}")

def calculate_rolling_revin(series, window=256, clip=3.0):
    """
    Calculates the 256-day Reversible Instance Normalization (Z-Score).
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero, fill early NaNs with the first available std
    rolling_std = rolling_std.replace(0, np.nan).bfill().fillna(1e-8)
    
    z_score = (series - rolling_mean) / rolling_std
    z_score = z_score.clip(-clip, clip)
    return z_score

def get_crack_spread(M=3):
    """
    Calculates the 3:2:1 crack spread for the M-th nearby contracts.
    """
    cl = build_continuous_futures('CL', M)
    rb = build_continuous_futures('RB', M)
    ho = build_continuous_futures('HO', M)
    
    # Align dates
    df = pd.concat([cl, rb, ho], axis=1).dropna()
    
    # Crack spread formula: (2 * RB * 42 + 1 * HO * 42) / 3 - CL
    crack = (2 * df[f'RB_M{M}'] * 42 + df[f'HO_M{M}'] * 42) / 3 - df[f'CL_M{M}']
    
    crack.name = f"Crack_321_M{M}"
    return crack

def get_equity_returns(tickers, weights=None, beta_window=60):
    """
    Calculates daily returns, B7 Basket returns, and 60-day rolling Beta against SPY.
    """
    df_list = []
    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")
        df = pd.read_csv(path, parse_dates=['date']).set_index('date')
        ret = df['Close'].pct_change().rename(ticker)
        df_list.append(ret)
        
    # Also load SPY for hedging
    spy = pd.read_csv(os.path.join(DATA_DIR, "SPY_daily.csv"), parse_dates=['date']).set_index('date')
    spy_ret = spy['Close'].pct_change().rename("SPY")
    df_list.append(spy_ret)
    
    ret_df = pd.concat(df_list, axis=1).dropna()
    
    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)
        
    basket_ret = pd.Series(0.0, index=ret_df.index)
    for t, w in zip(tickers, weights):
        basket_ret += ret_df[t] * w
        
    # Calculate rolling beta = Cov(Basket, SPY) / Var(SPY)
    cov = basket_ret.rolling(window=beta_window, min_periods=beta_window).cov(ret_df['SPY'])
    var = ret_df['SPY'].rolling(window=beta_window, min_periods=beta_window).var()
    beta = cov / var
    
    beta_hedged_ret = basket_ret - beta * ret_df['SPY']
    
    result = pd.DataFrame({
        'Basket_Return': basket_ret,
        'SPY_Return': ret_df['SPY'],
        'Rolling_Beta': beta,
        'Beta_Hedged_Return': beta_hedged_ret
    })
    
    # Include individual hedged returns as well for individual testing
    for t in tickers:
        t_cov = ret_df[t].rolling(window=beta_window, min_periods=beta_window).cov(ret_df['SPY'])
        t_beta = t_cov / var
        result[f'{t}_Hedged_Return'] = ret_df[t] - t_beta * ret_df['SPY']
        
    return result.dropna()

def build_master_dataset(M=3, beta_window=60):
    print("Building continuous crack spread (M{})...".format(M))
    crack = get_crack_spread(M)
    crack_z = calculate_rolling_revin(crack, window=256)
    crack_df = pd.DataFrame({
        'Crack_Spread': crack,
        'Crack_Z_Score': crack_z
    })
    
    print("Building equity basket and beta hedging...")
    tickers = ['VLO', 'MPC', 'PSX', 'DINO', 'PBF', 'DK', 'CVI']
    weights = [0.25, 0.25, 0.25, 0.10, 0.05, 0.05, 0.05]
    eq_df = get_equity_returns(tickers, weights, beta_window)
    
    print("Aligning final dataset...")
    final_df = pd.concat([crack_df, eq_df], axis=1).dropna()
    return final_df

if __name__ == "__main__":
    df = build_master_dataset(M=3, beta_window=60)
    print("\nSample Data (Last 5 days):")
    print(df[['Crack_Spread', 'Crack_Z_Score', 'Basket_Return', 'Beta_Hedged_Return']].tail())
    
    out_path = os.path.join(os.path.dirname(__file__), "master_dataset.csv")
    df.to_csv(out_path)
    print(f"\nSaved master dataset with {len(df)} rows to {out_path}")
