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
    features_per_day = 6  # Number of features per stock per day [close, open, high, low, volume, adjusted_close]

    # Extract close prices and volumes for calculating features
    prices = s[0::features_per_day]  # Close prices for all stocks
    volumes = s[4::features_per_day]  # Volume data for all stocks

    # Calculate daily log returns
    daily_returns = np.log(prices[1:] / prices[:-1])  # Daily log returns to ensure stability

    # Compute features for state representation
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)  # 3 scalars for 5, 10, 20 day momentum
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Create the updated state by concatenating original state with new features
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
    # Extract the key features for intrinsic reward calculation
    relative_momentum = updated_s[120]   # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility
    downside_risk = updated_s[122]        # Downside risk

    # Determine market regime and risk level (assumed dynamic in practice)
    market_regime = "Balanced"  # This should be dynamically set based on market analysis
    risk_level = 0.15  # This can be adjusted based on current market conditions

    # Initialize reward
    reward = relative_momentum * (1 - 0.5 * risk_level)  # Baseline reward

    # Modify reward based on market regime
    if market_regime in ["Aggressive", "Balanced"]:
        # Favorable conditions, encouraging exploration and capturing returns
        reward -= 0.5 * realized_volatility  # Penalty for increased volatility
    else:
        # Unfavorable conditions, applying penalties for volatility and additional downside risk
        penalty = max(0, realized_volatility - 0.025) + downside_risk  # Higher penalties in tougher regimes
        reward -= penalty

    return float(reward)