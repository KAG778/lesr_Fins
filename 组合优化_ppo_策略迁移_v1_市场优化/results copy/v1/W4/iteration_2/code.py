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
    num_assets = 5  # 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + 1 for CASH
    updated_features = []

    for i in range(num_assets):
        # Extract last 20 days of close prices and volumes for each stock
        start_index = i * 6  # Each stock represented in state s occupies 6 slots
        close_prices = s[start_index:start_index + 120:6]  # Get close prices
        volumes = s[start_index + 4:start_index + 120:6]  # Get volumes

        # Calculate additional features
        relative_momentum = compute_relative_momentum(close_prices)
        daily_returns = np.diff(close_prices) / close_prices[:-1]  # Daily return
        realized_volatility = compute_realized_volatility(daily_returns)
        downside_risk = compute_downside_risk(daily_returns)
        multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
        zscore_price = compute_zscore_price(close_prices)
        mean_reversion_strength = compute_mean_reversion_signal(close_prices)
        turnover_ratio = compute_turnover_ratio(volumes)

        # Append calculated features for this stock
        features = [
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_mom,
            zscore_price,
            mean_reversion_strength,
            turnover_ratio,
        ]
        
        updated_features.extend(features)

    # Concatenate the existing state with the new features
    updated_s = np.concatenate((s, updated_features))
    return updated_s

def intrinsic_reward(updated_s):
    risk_level = 0.07  # From market guidance
    relative_momentum = updated_s[120:125]  # Indices for relative momentum of 5 stocks
    realized_volatility = updated_s[125:130]  # Indices for realized volatility of 5 stocks
    downside_risk = updated_s[130:135]  # Indices for downside risk of 5 stocks

    if risk_level < 0.05:  # Favorable regime
        # Reward based on the average relative momentum adjusted for current risk level
        reward = np.mean(relative_momentum) * (1 - 0.5 * risk_level)
    else:
        # Penalize high volatility or downside risk
        high_volatility_penalty = np.mean(np.maximum(0, realized_volatility - 0.02))
        downside_risk_penalty = np.mean(np.maximum(0, downside_risk - 0.01))
        reward = - (high_volatility_penalty + downside_risk_penalty)

    return reward