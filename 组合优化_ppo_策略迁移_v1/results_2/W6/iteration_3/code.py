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
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Combine new features into a single array
    extra_features = np.array([
        relative_momentum,
        realized_volatility,
        downside_risk,
        multi_horizon_mom[0],  # 5-day momentum
        multi_horizon_mom[1],  # 10-day momentum
        multi_horizon_mom[2],  # 20-day momentum
        zscore_price,
        mean_reversion_signal,
        turnover_ratio
    ])
    
    # Concatenate original state with new features to create an updated state representation
    updated_s = np.concatenate((s, extra_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract important features from the revised state
    relative_momentum = updated_s[120]  # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility 
    downside_risk = updated_s[122]  # Downside risk
    risk_level = 0.33  # Example risk level from market regime; can be adjusted dynamically

    # Define thresholds for risk features to penalize excessive values
    VOLATILITY_THRESHOLD = 0.05  
    DOWNSIDE_RISK_THRESHOLD = 0.03  
    
    # Initialize intrinsic reward
    intrinsic_r = 0.0
    
    if risk_level > 0.5:  # In high-risk regimes
        # Penalize for high realized volatility and downside risk
        penalty = (np.maximum(realized_volatility - VOLATILITY_THRESHOLD, 0) +
                   np.maximum(downside_risk - DOWNSIDE_RISK_THRESHOLD, 0))
        intrinsic_r = -penalty  # Negative penalty to reduce risk exposure
    else:  # In more favorable market environments
        intrinsic_r = relative_momentum * (1 - 0.5 * risk_level)  # Encourage positive momentum for higher portfolio action
        
    return intrinsic_r