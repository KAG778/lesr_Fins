import numpy as np
from feature_library import (
    compute_relative_momentum,
    compute_realized_volatility,
    compute_downside_risk,
    compute_multi_horizon_momentum,
    compute_zscore_price,
    compute_mean_reversion_signal,
    compute_turnover_ratio
)

def revise_state(s):
    # Extract close prices and volumes from the state representation
    close_prices = s[0::6]  # Extract close prices for 20 days
    volumes = s[4::6]       # Extract volumes for 20 days
    
    # Calculate daily returns
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Calculate normalized daily returns
    
    # Calculate additional features using the provided feature library functions
    relative_momentum = compute_relative_momentum(close_prices)  # Relative momentum over 20 days
    realized_volatility = compute_realized_volatility(daily_returns)  # Realized volatility over 20 days
    downside_risk = compute_downside_risk(daily_returns)  # Downside risk over 20 days
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)  # Multi-horizon momentum
    zscore_price = compute_zscore_price(close_prices)  # Z-score of the current price
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)  # Mean reversion signal
    turnover_ratio = compute_turnover_ratio(volumes)  # Turnover ratio based on volume

    # Combine new features into a single array
    extra_features = np.array([
        relative_momentum,        # Feature 1: Relative momentum
        realized_volatility,      # Feature 2: Realized volatility
        downside_risk,           # Feature 3: Downside risk
        multi_horizon_mom[0],    # Feature 4: 5-day momentum
        multi_horizon_mom[1],    # Feature 5: 10-day momentum
        multi_horizon_mom[2],    # Feature 6: 20-day momentum
        zscore_price,             # Feature 7: Z-score of current price
        mean_reversion_signal,     # Feature 8: Mean reversion strength
        turnover_ratio             # Feature 9: Turnover ratio
    ])

    # Concatenate original state with new features to create an updated state representation
    updated_s = np.concatenate((s, extra_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract key features from the updated state
    relative_momentum = updated_s[120]    # Relative momentum
    realized_volatility = updated_s[121]   # Realized volatility 
    downside_risk = updated_s[122]         # Downside risk
    
    # Example risk_level based on market regime; can be dynamically adjusted in practice
    risk_level = 0.33  
        
    # Define thresholds for penalizing excessive risk
    VOLATILITY_THRESHOLD = 0.05  
    DOWNSIDE_RISK_THRESHOLD = 0.03  
    
    # Initialize intrinsic reward
    intrinsic_r = 0.0
    
    if risk_level > 0.5:  # In high-risk regimes
        # Penalize for high realized volatility and downside risk
        penalties = np.maximum(realized_volatility - VOLATILITY_THRESHOLD, 0) + \
                    np.maximum(downside_risk - DOWNSIDE_RISK_THRESHOLD, 0)
        intrinsic_r = -penalties  # Negative penalty to reduce risk exposure
    else:  # In balanced or favorable market environments
        intrinsic_r = relative_momentum * (1 - 0.5 * risk_level)  # Encourage positive momentum adjusted for risk level
        
    return intrinsic_r