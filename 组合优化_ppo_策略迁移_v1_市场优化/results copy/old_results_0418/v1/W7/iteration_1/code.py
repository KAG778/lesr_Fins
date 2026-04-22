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
    # Initialize the updated state with the original state
    updated_s = list(s)

    # Reshape state into days and channels
    closing_prices = s[0::6]  # closing prices
    volumes = s[4::6]         # volumes

    # Calculate additional features
    relative_momentum = compute_relative_momentum(closing_prices)
    realized_volatility = compute_realized_volatility(np.diff(np.log(closing_prices)))
    downside_risk = compute_downside_risk(np.diff(np.log(closing_prices)))
    multi_horizon_momentum = compute_multi_horizon_momentum(closing_prices)
    zscore_price = compute_zscore_price(closing_prices)
    mean_reversion_signal = compute_mean_reversion_signal(closing_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Concatenate newly calculated features to the state
    updated_s.extend([
        relative_momentum,        # Additional feature
        realized_volatility,      # Additional feature
        downside_risk,           # Additional feature
        *multi_horizon_momentum,  # Three additional features (5, 10, 20 days)
        zscore_price,            # Additional feature
        mean_reversion_signal,    # Additional feature
        turnover_ratio            # Additional feature
    ])

    # Convert updated_s back to a numpy array
    return np.array(updated_s)

def intrinsic_reward(updated_s):
    # Extract computed features from updated state
    relative_momentum = updated_s[120]
    realized_volatility = updated_s[121]
    downside_risk = updated_s[122]
    
    # Determine the current market regime
    market_regime = "Balanced"  # This should ideally be dynamic based on market conditions

    # Define reward thresholds for the intrinsic reward calculation
    volatility_threshold = 0.05  # Example threshold for penalty
    downside_risk_threshold = 0.03  # Example threshold for downside risk

    if market_regime in ["Defensive", "Crisis"]:
        # Penalize high volatility and downside risk in crisis mode
        reward = -0.5 * max(0, realized_volatility - volatility_threshold) \
                 - 0.5 * max(0, downside_risk - downside_risk_threshold)
    else:
        # Favor exploration in balanced markets with incentives for relative momentum
        reward = relative_momentum

    return reward