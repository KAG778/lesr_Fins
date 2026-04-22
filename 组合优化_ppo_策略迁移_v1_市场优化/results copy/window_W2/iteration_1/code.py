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
    # Reshape the state array into a 2D array (20 days x 6 features)
    prices = s[0::6]  # Close prices
    volumes = s[4::6]  # Volumes
    
    # Calculate various features
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(np.diff(prices) / prices[:-1])
    downside_risk = compute_downside_risk(np.diff(prices) / prices[:-1])
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Concatenate these new features to the original state
    updated_s = np.concatenate((s, 
                                 [relative_momentum, 
                                  realized_volatility, 
                                  downside_risk] + list(multi_horizon_momentum) + 
                                 [zscore_price, 
                                  mean_reversion_signal, 
                                  turnover_ratio]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract features and original dimensions
    risk_feature = updated_s[120]  # Realized volatility (1D scalar)
    downside_risk = updated_s[121]  # Downside risk (1D scalar)
    trend_signal = updated_s[122]  # Relative momentum (1D scalar)
    
    # Define thresholds based on market guidance
    volatility_threshold = 0.05  # High risk threshold (5% daily vol as an example)
    
    # Market regime assessment - assume this is determined somewhere in the environment
    market_regime = "Balanced"  # For the sake of example; this can vary.

    if market_regime in ["Defensive", "Crisis"]:
        # Penalize high risk states
        intrinsic_reward = -max(0, risk_feature - volatility_threshold) - max(0, downside_risk)
    else:
        # Reward exploration in a balanced market
        intrinsic_reward = trend_signal * (1 - 0.5 * (downside_risk / max(1e-5, downside_risk)))  # Normalize downside risk
    
    return intrinsic_reward