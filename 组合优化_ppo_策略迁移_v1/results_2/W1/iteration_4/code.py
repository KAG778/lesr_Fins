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
    features_per_day = 6  # Close, open, high, low, volume, adjusted_close

    # Extract the latest close prices and volumes for feature calculations
    prices = s[0::features_per_day]  # Close prices for all assets
    volumes = s[4::features_per_day]  # Volume data for all assets

    # Calculate daily log returns for the last 20 days
    daily_returns = np.log(prices[1:] / prices[:-1])  # Daily log returns for stability

    # Compute additional market indicators
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)  # 5, 10, and 20 day momentum
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Combine original state with the new features to create an updated state
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
    # Extract key features from the updated state representation
    relative_momentum = updated_s[120]   # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility
    downside_risk = updated_s[122]        # Downside risk

    # Define simplified market regime and risk level for the intrinsic reward calculation
    market_regime = "Balanced"  # This should ideally be dynamically evaluated from market data
    risk_level = 0.15  # Risk level parameter based on market conditions, defaulting per guidance
    stable_bonus = 0.1  # A small bonus for stability consideration

    # Compute the intrinsic reward based on current market conditions
    if market_regime in ["Aggressive", "Balanced"]:
        # Favorable conditions, encouraging exploration of wealth-generating states
        reward = relative_momentum * (1 - 0.5 * risk_level) + stable_bonus
    else:
        # Unfavorable conditions, applying penalties for high volatility
        penalty = max(0, realized_volatility - 0.025) + downside_risk  # High penalty for excessive volatility
        reward = relative_momentum - penalty  # Adjust reward to penalize riskier actions

    return reward