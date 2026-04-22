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
    # Extract close prices and volumes from the original state
    close_prices = s[0::6]      # Retrieve close prices from the state
    volumes = s[4::6]           # Retrieve volumes from the state

    # Calculate returns from close prices (daily returns)
    # Avoid using the first element to prevent overflow issues
    returns = np.diff(close_prices) / close_prices[:-1]

    # Calculate new features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Combine the original state with the new features
    updated_s = np.concatenate([
        s,                       # Original state (120 dimensions)
        np.array([relative_momentum, realized_volatility, downside_risk]), # 3 additional features
        multi_horizon_momentum,  # 3 more features from multi-horizon momentum
        np.array([zscore_price, mean_reversion_signal, turnover_ratio])  # Final 3 features
    ])

    return updated_s

def intrinsic_reward(updated_s):
    # Extract new features
    relative_momentum = updated_s[120]  # assuming this is relative_momentum
    realized_volatility = updated_s[121]  # assuming this is realized_volatility
    downside_risk = updated_s[122]  # assuming this is downside_risk

    # Define market regimes using external data (can be improved dynamically)
    market_regime = "Balanced"  # Placeholder for current market regime determination

    # Calculate intrinsic reward based on market conditions
    # Market Strategy Guidance
    if realized_volatility > 0.04 or downside_risk > 0.02:  # Example thresholds
        # Penalize for high realized volatility or downside risk
        intrinsic_reward = -1 * (realized_volatility + downside_risk) 
    else:
        # Encourage exploration of momentum and penalize downside risks
        intrinsic_reward = relative_momentum * (1 - 0.5 * (realized_volatility + downside_risk))

    return intrinsic_reward