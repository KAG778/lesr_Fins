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
    # Reshape input state for easier access
    close_prices = s[0::6]
    volumes = s[4::6]
    
    # Compute necessary features
    relative_momentum = np.array([compute_relative_momentum(close_prices[i:i + 20]) for i in range(len(close_prices) - 19)])[-1]
    realized_volatility = compute_realized_volatility(np.diff(np.log(close_prices)), window=20)
    downside_risk = compute_downside_risk(np.diff(np.log(close_prices)), window=20)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices, window=20)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices, window=20)
    turnover_ratio = compute_turnover_ratio(volumes, window=20)
    
    # Update state with computed features by concatenating them
    extra_features = np.concatenate(([relative_momentum, realized_volatility, downside_risk],
                                      multi_horizon_momentum,
                                      [zscore_price, mean_reversion_signal],
                                      [turnover_ratio]))
    
    updated_s = np.concatenate((s, extra_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract features from the updated state
    realized_volatility = updated_s[120]
    downside_risk = updated_s[121]
    turnover_ratio = updated_s[126]
    
    # Constants for reward calculation
    volatility_threshold = 0.04  # Using 4% as a hypothetical threshold for high volatility
    
    # Intrinsic reward based on market regime
    current_market_regime = 'Balanced'  # Assuming this is determined outside and passed in
    if current_market_regime in ['Defensive', 'Crisis']:
        # Penalizing high volatility and downside risk in defensive/crisis regimes
        reward = -1 * max(0, realized_volatility - volatility_threshold)
    else:  # Favorable regimes focus on exploration
        reward = (1 - 0.5 * downside_risk) + 0.1 * turnover_ratio  # Example balance
    
    return reward