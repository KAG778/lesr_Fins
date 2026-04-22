import numpy as np
from feature_library import (compute_relative_momentum,
                              compute_realized_volatility,
                              compute_downside_risk,
                              compute_multi_horizon_momentum,
                              compute_zscore_price,
                              compute_mean_reversion_signal,
                              compute_turnover_ratio)

def revise_state(s):
    # Extracting the 20 days of close prices and volumes for calculations
    close_prices = s[0::6]  # Close prices
    volumes = s[4::6]        # Volumes
    
    # Calculate additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(np.diff(close_prices))  # Daily returns
    downside_risk = compute_downside_risk(np.diff(close_prices))  # Daily returns
    multi_horizon_mo = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Create updated state
    updated_s = np.concatenate([
        s,                                 # original state
        np.array([relative_momentum, realized_volatility, downside_risk]),  # risk and momentum features
        multi_horizon_mo,                 # multi-horizon momentum (3 features)
        np.array([zscore_price, mean_reversion_signal, turnover_ratio])  # additional features
    ])
    
    return updated_s

def intrinsic_reward(updated_s):
    # Use specific dimensions in the updated state for reward calculation
    realized_volatility = updated_s[120]  # Adjusted to reflect position in updated state
    downside_risk = updated_s[121]
    relative_momentum = updated_s[122]  # First extra feature from updated state

    current_market_regime = "moderate"  # Can be changed based on the environment's state

    alpha = 1  # Risk aversion parameter

    if current_market_regime == "balanced":
        # Encourage exploration by rewarding higher relative momentum
        reward = relative_momentum
    else:
        # Penalize high risk in unfavorable regimes
        risk_threshold = 0.02  # Define a threshold for the penalty
        reward = -alpha * max(0, realized_volatility - risk_threshold)

    return reward