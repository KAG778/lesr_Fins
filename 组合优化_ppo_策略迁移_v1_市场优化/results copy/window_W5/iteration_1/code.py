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
    # Extract 20 days' close prices and volumes
    close_prices = s[0::6]  # s[0], s[6], ..., s[114]
    volumes = s[4::6]       # s[4], s[10], ..., s[119]

    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1]  # daily returns

    # Compute additional features
    relative_momentum = np.array([compute_relative_momentum(close_prices)])
    realized_volatility = np.array([compute_realized_volatility(returns)])
    downside_risk = np.array([compute_downside_risk(returns)])
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    zscore_price = np.array([compute_zscore_price(close_prices)])
    mean_reversion_signal = np.array([compute_mean_reversion_signal(close_prices)])
    turnover_ratio = np.array([compute_turnover_ratio(volumes)])

    # Prepare the updated state
    updated_s = np.concatenate((s, 
                                 relative_momentum, 
                                 realized_volatility, 
                                 downside_risk, 
                                 multi_horizon_momentum, 
                                 zscore_price, 
                                 mean_reversion_signal, 
                                 turnover_ratio))

    return updated_s


def intrinsic_reward(updated_s):
    # Extract source dimensions
    realized_volatility = updated_s[120]  # Where additional features start
    downside_risk = updated_s[121]        # Next feature
    zscore_price = updated_s[126]         # Next feature
    reward_signal = updated_s[0:120]      # Original state information

    # Define the market regime focus
    # For demonstration, we'll assume a defensive regime (this should be based on external regime detection)
    is_defensive_regime = True  # In a real scenario, determine this externally

    # Define parameters
    epsilon = 1e-10  # Small constant to avoid division by zero
    alpha = 1.0  # Penalty multiplier for risk features

    if is_defensive_regime:
        # Penalizing high risk volatility
        intrinsic_r = -alpha * max(0, realized_volatility + downside_risk - 0.05)  # Assuming 0.05 as a threshold
        return intrinsic_r
    else:
        # Encouragement signal for exploration of informative features
        # This is a placeholder; user-defined logic would go here based on the context of favorable regimes
        exploration_signal = np.mean(reward_signal)  # A simple mean reward exploration
        return exploration_signal