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
    num_assets = 5  # 5 stocks (excluding CASH)
    features_per_day = 6  # [close, open, high, low, volume, adjusted_close]
    
    # Extract relevant price and volume data from the state representation
    prices = s[0::features_per_day]  # Close prices for all assets
    volumes = s[4::features_per_day]  # Volume data for all assets
    
    # Calculate returns as log differences for stability
    daily_returns = np.log(prices[1:] / prices[:-1])  # Daily log returns

    # Compute features to enhance state representation
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)  # 3 values for 5, 10, 20 day momentum
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Combine original state with the new features
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
    # Extract necessary features from the updated state representation
    relative_momentum = updated_s[120]   # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility
    downside_risk = updated_s[122]        # Downside risk
    
    # Market regime parameters based on analysis
    market_regime = "Balanced"  # This would ideally be dynamically assessed
    risk_level = 0.15  # Risk level parameter based on market conditions
    penalty_threshold = 0.025  # Define a threshold for high penalty

    # Compute the intrinsic reward based on market conditions
    if market_regime in ["Aggressive", "Balanced"]:
        reward = relative_momentum - 0.5 * realized_volatility  # Encourage exploration in favorable market conditions
    else:
        penalty = max(0, realized_volatility - penalty_threshold) + downside_risk  # Penalize high volatility
        reward = relative_momentum - penalty  # Adjust reward for possibly unfavorable conditions

    return reward