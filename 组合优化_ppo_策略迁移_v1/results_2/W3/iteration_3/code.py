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
    
    # Prepare to collect all feature signals for the updated state
    extended_features = []
    
    for asset_idx in range(num_assets):
        close_prices = s[asset_idx::features_per_asset][:20]  # Close prices for the last 20 days
        volumes = s[4 + asset_idx * features_per_asset::features_per_asset][:20]  # Volumes for the last 20 days
        
        # Calculate additional features
        relative_momentum = compute_relative_momentum(close_prices)
        realized_volatility = compute_realized_volatility(np.diff(close_prices))
        downside_risk = compute_downside_risk(np.diff(close_prices))
        multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
        zscore_price = compute_zscore_price(close_prices)
        mean_reversion_signal = compute_mean_reversion_signal(close_prices)
        turnover_ratio = compute_turnover_ratio(volumes)
        
        # Append derived features to the extended feature list
        extended_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_mom,  # appending three values from multi-horizon momentum
            zscore_price,
            mean_reversion_signal,
            turnover_ratio,
        ])
    
    # Construct the new state with original and extended features
    updated_s = np.concatenate((s, np.array(extended_features)))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant derived features
    relative_momentum = updated_s[120]  # Assuming this is where relative momentum is located
    realized_volatility = updated_s[121] # Assuming this is the realized volatility
    downside_risk = updated_s[122]       # Assuming this is the downside risk
 
    # Define reward based on the market conditions
    # Distinguish between high-risk and favorable regimes
    high_risk_threshold = 0.04  # Example threshold for high-risk regime
    if realized_volatility > high_risk_threshold:
        # In a high-risk environment, penalize for downside risk
        reward = -1 * max(0, downside_risk)  # Penalize if downside risk is significant
    else:
        # Favorable regime: reward for relative momentum adjusted by the downside risk
        reward = relative_momentum * (1 - 0.5 * downside_risk)  # Encourage exploration

    # Normalize or constrain the reward
    reward = np.clip(reward, -10, 10)  # Clip the reward to avoid extreme values

    return reward