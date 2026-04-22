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
    num_assets = 6  # 5 stocks + 1 CASH
    features_per_day = 6  # [close, open, high, low, volume, adjusted_close]
    
    # Initialize a list to hold the extended features
    extended_features = []

    for asset_idx in range(num_assets):
        close_prices = s[asset_idx::features_per_day][:20]  # Get close prices for this asset
        volumes = s[4 + asset_idx * features_per_day::features_per_day][:20]  # Get volumes for this asset
        
        # Compute required features
        relative_momentum = compute_relative_momentum(close_prices)
        realized_volatility = compute_realized_volatility(np.diff(close_prices))
        downside_risk = compute_downside_risk(np.diff(close_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
        z_score = compute_zscore_price(close_prices)
        mean_reversion_signal = compute_mean_reversion_signal(close_prices)
        turnover_ratio = compute_turnover_ratio(volumes)
        
        # Extend the feature list with computed values
        extended_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,  # Add multi-horizon momentum results
            z_score,
            mean_reversion_signal,
            turnover_ratio,
        ])
    
    # Concatenate the original state with the new features
    updated_s = np.concatenate((s, np.array(extended_features)))
    return updated_s

def intrinsic_reward(updated_s):
    # Extract computed features for reward calculation
    relative_momentum = updated_s[120]  # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility
    downside_risk = updated_s[122]  # Downside risk
    
    # Define reward based on the current market conditions
    # Let's assume a simple regime detection mechanism based on a threshold
    high_volatility_threshold = 0.04  # Adjust these thresholds based on empirical analysis
    
    if realized_volatility > high_volatility_threshold:
        # In a high-risk environment, we apply a penalty to the reward
        reward = -1 * max(0, downside_risk)  # Penalize for downside risk
    else:
        # Favorable regime, reward based on relative momentum while adjusting for risk
        reward = relative_momentum * (1 - 0.5 * downside_risk)  # Balanced reward

    # Clip the reward to maintain stability
    return np.clip(reward, -10, 10)  # Helipping to manage extreme reward values