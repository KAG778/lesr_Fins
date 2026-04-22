import numpy as np
from feature_library import (compute_relative_momentum,
                              compute_realized_volatility,
                              compute_downside_risk,
                              compute_multi_horizon_momentum,
                              compute_zscore_price,
                              compute_mean_reversion_signal,
                              compute_turnover_ratio)

def revise_state(s):
    # Extract close prices and volume data from the state s
    close_prices = s[0::6]  # close prices for 20 days
    volumes = s[4::6]       # volumes for 20 days
    
    # Calculate additional features
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Calculate daily returns
    
    relative_momentum = compute_relative_momentum(close_prices)  # Excess return vs window-average
    realized_volatility = compute_realized_volatility(daily_returns)  # Realized volatility
    downside_risk = compute_downside_risk(daily_returns)  # Downside risk
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)  # Momentum at multiple horizons
    zscore_price = compute_zscore_price(close_prices)  # Z-score of current price vs N-day mean
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)  # Mean reversion strength
    turnover_ratio = compute_turnover_ratio(volumes)  # Current volume / average volume
    
    # Concatenate new features to the original state
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
    
    updated_s = np.concatenate((s, extra_features))  # Combine original state and new features
    return updated_s

def intrinsic_reward(updated_s):
    # Extract necessary features from the updated state
    realized_volatility = updated_s[120]  # Realized Volatility from the revised state
    downside_risk = updated_s[121]  # Downside risk from the revised state
    relative_momentum = updated_s[122]  # Relative momentum from the revised state

    # Market regime parameters
    risk_level = 0.33  # From market strategy guidance (current regime is balanced)

    # Define thresholds for risk features
    volatility_threshold = 0.02  # Arbitrary threshold for volatility; can be tuned based on analysis
    downside_threshold = 0.01  # Arbitrary threshold for downside risk; can be tuned based on analysis

    if risk_level > 0.5:  # If risk level is high, we want to penalize
        # Penalize for high realized volatility and downside risk
        penalty = -1.0 * (max(0, realized_volatility - volatility_threshold) + 
                          max(0, downside_risk - downside_threshold))
    else:
        # Reward for beneficial feature exploration in a balanced regime
        reward = relative_momentum  # Explore informative feature like momentum
        penalty = 0

    intrinsic_reward = penalty + reward
    return intrinsic_reward