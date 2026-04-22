import numpy as np
from feature_library import (compute_relative_momentum, 
                              compute_realized_volatility, 
                              compute_downside_risk, 
                              compute_multi_horizon_momentum, 
                              compute_zscore_price, 
                              compute_mean_reversion_signal, 
                              compute_turnover_ratio)

def revise_state(s):
    # Extract close prices and volumes from the original state
    close_prices = s[0::6]      # 20 close prices (s[0], s[6], ..., s[114])
    volumes = s[4::6]           # 20 volumes (s[4], s[10], ..., s[114])

    # Calculate returns from close prices (daily returns)
    # Avoid using the first element to prevent overflow issues
    returns = np.diff(close_prices) / close_prices[:-1]

    # Calculate new features
    momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    
    # Multi-horizon momentum
    multi_momentum = compute_multi_horizon_momentum(close_prices)

    # Z-score of the latest price relative to 20-day mean
    zscore_price = compute_zscore_price(close_prices)

    # Mean reversion signal
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)

    # Turnover ratio
    turnover_ratio = compute_turnover_ratio(volumes)

    # Combine original state with new features
    updated_s = np.concatenate([
        s,                       # Original state
        np.array([momentum, realized_volatility, downside_risk]), # New features for individual stock
        multi_momentum,         # 3 additional features from the multi-horizon momentum
        np.array([zscore_price, mean_reversion_signal, turnover_ratio])  # 3 more features
    ])

    return updated_s


def intrinsic_reward(updated_s):
    # Extract additional features
    realized_volatility = updated_s[120]  # new feature from revised state
    downside_risk = updated_s[121]         # new feature from revised state

    # Define thresholds based on market conditions (for example, we will define them arbitrarily)
    # In a real implementation, these threshold values should be derived from market regime indicators
    volatility_threshold = 0.035  # Example threshold for high risk
    penalty_factor = 10  # To adjust the penalty strength

    # Calculate reward based on market conditions
    if realized_volatility > volatility_threshold or downside_risk > volatility_threshold:
        # Penalize high risk states
        reward = -penalty_factor * max(0, realized_volatility - volatility_threshold)
    else:
        # Given the balanced market conditions, we want to reward exploration
        reward = 0.5 * (1 - realized_volatility)

    return reward