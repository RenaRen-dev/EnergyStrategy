import numpy as np
import pandas as pd

def apply_risk_defense(target_size, ticker, history_T, current_notional):
    """
    Applies the Three-Layer Risk Defense System.
    """
    # Defensive data needed
    spy_ret_series = history_T["SPY_Return"]
    
    # Calculate 21-day annualized volatility, defaulting to 0.15 if not enough history
    if len(history_T) >= 21:
        spy_vol = spy_ret_series.iloc[-21:].std() * np.sqrt(252)
    else:
        spy_vol = 0.15
        
    # Layer 1: Volatility Scaling (De-leveraging)
    target_vol = 0.15
    vol_scalar = min(1.0, target_vol / max(spy_vol, 0.05))
    target_size *= vol_scalar
    
    # Layer 2: Correlation Break-Detector (Herding/Liquidity Crisis)
    # Check correlation between asset and SPY over last 21 days
    asset_ret_col = f"{ticker}_Hedged_Return"
    if asset_ret_col in history_T.columns and len(history_T) >= 21:
        asset_ret = history_T[asset_ret_col].iloc[-21:]
        corr = asset_ret.corr(spy_ret_series.iloc[-21:])
        if pd.notna(corr) and corr > 0.8:
            target_size = 0.0  # Force flat position if correlated panic is detected
            
    # Layer 3: Tail-Risk Hedge (Dynamic Directional Switch)
    # This logic returns a tuple (new_target_size, hedge_size)
    hedge_size = 0.0
    if spy_vol > 0.30:
        # Emergency Hedge: Short SPY if market is in panic
        hedge_size = current_notional * -1.5 
        target_size = 0.0 # Clear primary exposure
        
    return target_size, hedge_size
