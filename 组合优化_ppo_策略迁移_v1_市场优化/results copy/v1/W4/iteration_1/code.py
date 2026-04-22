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
    # Extract prices and volumes from the input state array
    close_prices = s[0::6]  # Close prices
    volumes = s[4::6]        # Volume
    
    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1]  # Daily returns based on close prices
    
    # Compute feature enhancements based on the calculations provided
    relative_momentum = np.array([compute_relative_momentum(close_prices[i:i+20]) for i in range(len(close_prices) - 20 + 1)])
    realized_volatility = compute_realized_volatility(returns[-20:], window=20)
    downside_risk = compute_downside_risk(returns[-20:], window=20)
    
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices, windows=[5, 10, 20])
    zscore_price = compute_zscore_price(close_prices[-20:], window=20)
    mean_reversion_strength = compute_mean_reversion_signal(close_prices[-20:], window=20)
    turnover_ratio = compute_turnover_ratio(volumes[-20:], window=20)

    # Concatenate new features to the original state representation
    updated_s = np.concatenate((
        s,
        np.array([
            relative_momentum[-1],  # Last relative momentum
            realized_volatility,
            downside_risk,
            *multi_horizon_mom,
            zscore_price,
            mean_reversion_strength,
            turnover_ratio
        ])
    ))

    return updated_s

def intrinsic_reward(updated_s):
    # Assume we are currently in a balanced market regime.
    risk_level = 0.07  # From market guidance
    excess_risk = updated_s[120]  # Using realized volatility from the updated state
    downside_risk = updated_s[121]  # Using downside risk from the updated state

    # Intrinsic reward based on the risk level and the desired features
    if risk_level > 0.05:  # If market is showing high-risk tendencies
        # Penalize for high volatility or downside risk
        reward = -0.1 * max(0, excess_risk - 0.02) - 0.2 * max(0, downside_risk - 0.01)
    else:
        # In a balanced regime, encourage exploration of features
        trend_signal = updated_s[120]  # Using relative momentum
        reward = trend_signal * (1 - 0.5 * risk_level)

    return reward