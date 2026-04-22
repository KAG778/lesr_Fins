import numpy as np
from feature_library import (
    compute_relative_momentum,
    compute_realized_volatility,
    compute_downside_risk,
    compute_multi_horizon_momentum,
    compute_zscore_price,
    compute_mean_reversion_signal,
    compute_turnover_ratio,
)

def revise_state(s):
    # Extract prices (close prices) and volumes from the state representation
    close_prices = s[0::6]  # Extract close prices from the state
    volumes = s[4::6]       # Extract volume data from the state

    # Calculate daily returns from close prices
    returns = np.diff(close_prices) / close_prices[:-1]
    # Handling the case where returns could be empty (only happens when less than 2 days of prices are available)
    if returns.size == 0:
        returns = np.zeros(20)  # Assign zeros if no returns can be calculated

    # Compute additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    z_score = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Construct the updated state with additional features
    updated_s = np.concatenate([
        s,
        np.array([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            z_score,
            mean_reversion_signal,
            turnover_ratio
        ])
    ])

    return updated_s

def intrinsic_reward(updated_s):
    # Extract risk and momentum features from the updated state
    realized_volatility = updated_s[120]  # Realized Volatility
    downside_risk = updated_s[121]         # Downside Risk
    relative_momentum = updated_s[126]     # Relative Momentum
    z_score = updated_s[124]                # Z-score for assessing price dynamics

    # Determine the current market regime: assuming we can get this dynamically in practice
    market_regime = 'Defensive'  # For this implementation; can be dynamic in actual setup

    # Set thresholds for risk assessment (can be adjusted based on strategies or historical performance)
    threshold_volatility = 0.04  # Adjusted threshold for realized volatility
    threshold_downside = 0.03     # Adjusted threshold for downside risk
    epsilon = 1e-8  # Small value to prevent division by zero

    if market_regime == 'Defensive':
        # Penalization for elevated risks in a defensive market
        risk_penalty = max(0, realized_volatility - threshold_volatility) + max(0, downside_risk - threshold_downside)
        reward = -risk_penalty  # Negative reward for higher risks

        # Reward for potential mean-reversion if z-score indicates it
        if z_score < 0:  # Indicating potential mean-reversion
            reward += abs(z_score) * 10  # Scale reward positively based on z-score

    else:
        # Reward based on positive relative momentum in a favorable market regime
        reward = relative_momentum * 100  # Scale momentum for better exploration

    return float(reward)  # Return the reward as a float for compatibility