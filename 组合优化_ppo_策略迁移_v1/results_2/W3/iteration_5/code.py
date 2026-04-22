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
    features_per_asset = 6  # [close, open, high, low, volume, adjusted_close]
    num_days = 20  # Historical data for 20 days

    # Initialize a list to hold computed features
    extended_features = []

    for asset_idx in range(num_assets):
        close_prices = s[asset_idx::features_per_asset][:num_days]  # Get close prices
        volumes = s[4 + asset_idx * features_per_asset::features_per_asset][:num_days]  # Get volumes

        # Calculate additional features
        relative_momentum = compute_relative_momentum(close_prices)
        realized_volatility = compute_realized_volatility(np.diff(close_prices))
        downside_risk = compute_downside_risk(np.diff(close_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
        z_score = compute_zscore_price(close_prices)
        mean_reversion_signal = compute_mean_reversion_signal(close_prices)
        turnover_ratio = compute_turnover_ratio(volumes)

        # Append features to the list
        extended_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,  # Multi-horizon momentum returns 3 values
            z_score,
            mean_reversion_signal,
            turnover_ratio,
        ])
    
    # Concatenate the original state with the new features
    updated_s = np.concatenate((s, np.array(extended_features)))

    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features for calculating the reward
    relative_momentum = updated_s[120]  # Relative momentum feature
    realized_volatility = updated_s[121]  # Realized volatility feature
    downside_risk = updated_s[122]        # Downside risk feature

    # Define threshold for high-risk regime
    high_risk_threshold = 0.03  # Example threshold for high-risk regime

    # Calculate reward based on current market conditions
    if realized_volatility > high_risk_threshold:
        # Penalize in high-risk regime
        reward = -1 * (downside_risk + realized_volatility)  # Combine penalties for downside risk and realized volatility
    else:
        # Reward in favorable regime
        reward = relative_momentum * (1 - 0.5 * downside_risk)  # Favor exploration weighted by downside risk reduction

    # Normalize or constrain the reward to mitigate extreme values
    return np.clip(reward, -10, 10)  # Clipping to prevent instability