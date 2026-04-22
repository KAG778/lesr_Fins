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
    # Reshape the state array for easier processing
    prices = s[0::6]  # Close prices
    volumes = s[4::6]  # Volume data

    # Calculate derived features 
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(np.log(prices[1:] / prices[:-1]))
    downside_risk = compute_downside_risk(np.log(prices[1:] / prices[:-1]))
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Combine all the features into a new state
    updated_s = np.concatenate([
        s, 
        np.array([
            relative_momentum,       # 1 scalar
            realized_volatility,     # 1 scalar
            downside_risk,          # 1 scalar
            *multi_horizon_momentum, # 3 scalars
            zscore_price,           # 1 scalar
            mean_reversion_signal,   # 1 scalar
            turnover_ratio           # 1 scalar
        ])
    ])
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract original state dimensions and additional features
    relative_momentum = updated_s[120]
    realized_volatility = updated_s[121]
    downside_risk = updated_s[122]

    # Define the market regime parameters
    risk_level = 0.15  # From market strategy guidance, assuming it remains constant
    
    # If the market is favorable, reward exploration
    reward = relative_momentum - 0.5 * (realized_volatility + downside_risk)
    
    # Penalize for high volatility/downside risk if risk level is high
    if risk_level > 0.10:  # Assuming thresholds, can be adjusted
        penalty = max(0, realized_volatility - 0.04)  # Example penalty threshold
        reward -= penalty
        
    return reward