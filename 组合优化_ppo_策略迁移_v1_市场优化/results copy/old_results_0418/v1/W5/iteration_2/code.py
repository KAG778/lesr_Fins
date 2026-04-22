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
    # Extract close prices and volumes from the original state representation
    close_prices = s[0::6]  # Every 6th dimension starting from index 0 gives close prices
    volumes = s[4::6]       # Every 6th dimension starting from index 4 gives volumes

    # Calculate features
    relative_momentum = compute_relative_momentum(close_prices)
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Daily returns calculation

    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices, windows=[5, 10, 20])
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Concatenate the new features with the original state
    updated_s = np.concatenate(
        (s, 
         np.array([relative_momentum, realized_volatility, downside_risk] + list(multi_horizon_momentum) + 
                   [zscore_price, mean_reversion_signal, turnover_ratio]))
    )
    return updated_s

def intrinsic_reward(updated_s):
    # Define constants and thresholds for reward calculation
    alpha = 1.0
    high_volatility_threshold = 0.05
    downside_risk_threshold = 0.02

    # Extract features related to risk from the updated state
    realized_volatility = updated_s[120]  # derived from revise_state
    downside_risk = updated_s[121]        # derived from revise_state
    mean_reversion_signal = updated_s[123] # index of the mean reversion signal

    # Current market regime can be dynamically set, here assumed as "Defensive"
    current_regime = "Defensive"

    if current_regime == "Defensive":
        # Penalize high volatility and downside risk in a defensive market regime
        risk_penalty = (max(0, realized_volatility - high_volatility_threshold) + 
                        max(0, downside_risk - downside_risk_threshold))
        reward = -alpha * risk_penalty
    else:
        # In less risky favorable conditions, reward informative features
        reward = mean_reversion_signal  # Reward based on the strength of mean reversion

    return reward