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
    # Reshape the state to easily manipulate per stock data
    reshaped_s = s.reshape(20, 6)  # 20 days * 6 features

    # Prepare a list to hold the additional features
    additional_features = []

    # Iterate through daily data
    close_prices = reshaped_s[:, 0]  # Close prices for all days
    volumes = reshaped_s[:, 4]        # Volumes for all days

    # Calculate features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(np.diff(close_prices) / close_prices[:-1])
    downside_risk = compute_downside_risk(np.diff(close_prices) / close_prices[:-1])

    # Multi-Horizon Momentum
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    
    # Z-Score Price
    zscore_price = compute_zscore_price(close_prices)
    
    # Mean Reversion Signal
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    
    # Turnover Ratio
    turnover_ratio = compute_turnover_ratio(volumes)

    # Append calculated features to the additional features list
    additional_features.extend([
        relative_momentum,
        realized_volatility,
        downside_risk,
        *multi_horizon_momentum,
        zscore_price,
        mean_reversion_signal,
        turnover_ratio
    ])
    
    # Append additional features to the original state and return
    updated_s = np.concatenate((s, additional_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract features from the updated state
    realized_volatility = updated_s[120]  # Assuming index for realized volatility
    downside_risk = updated_s[121]         # Assuming index for downside risk
    trend_signal = updated_s[122]          # Assuming index for relative momentum
    risk_level = 0.02  # Example risk level from market guidance; this can be from an external source
    market_regime = "Balanced"  # Example market regime; this should be set based on external logic
    
    # Initialize reward variable
    intrinsic_r = 0.0

    # Determine reward based on market regime
    if market_regime in ["Crisis", "Defensive"]:
        # Penalize high-risk states
        intrinsic_r = -max(0, realized_volatility - 0.05) - max(0, downside_risk)  # Adjusted threshold
    else:
        # Favor exploration in a favorable regime
        intrinsic_r = trend_signal * (1 - 0.5 * (downside_risk / max(1e-5, downside_risk)))  # Normalize downside risk

    return intrinsic_r