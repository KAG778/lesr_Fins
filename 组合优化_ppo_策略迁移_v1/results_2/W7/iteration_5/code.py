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
    num_stocks = 5
    additional_features = []

    for i in range(num_stocks):
        # Extract prices and volumes for the current stock
        close_prices = s[i * 6:i * 6 + 120:6]  # close prices for 20 days
        volumes = s[i * 6 + 4:i * 6 + 120 + 4:6]  # volumes for 20 days

        # Compute various features
        rel_momentum = compute_relative_momentum(close_prices)
        daily_returns = np.diff(close_prices) / close_prices[:-1]  # calculate returns
        realized_volatility = compute_realized_volatility(daily_returns)
        downside_risk = compute_downside_risk(daily_returns)
        multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
        zscore = compute_zscore_price(close_prices)
        mean_reversion = compute_mean_reversion_signal(close_prices)
        turnover = compute_turnover_ratio(volumes)

        # Aggregate features for this stock
        additional_features.extend([
            rel_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            zscore,
            mean_reversion,
            turnover
        ])

    # Convert additional features to numpy array
    additional_features = np.array(additional_features)

    # Combine with the original state
    updated_state = np.concatenate((s, additional_features))
    
    return updated_state

def intrinsic_reward(updated_s):
    # Extract features from updated state
    rel_momentum = updated_s[120:125]  # relative momentum for each stock
    realized_volatility = updated_s[125:130]  # realized volatility for each stock
    downside_risk = updated_s[130:135]  # downside risk for each stock

    # Fetch risk level from market guidance
    risk_level = 0.14  # Given risk level from market guidance
    risk_threshold = 0.03  # Example threshold for high risk
    alpha = 0.01  # Coefficient to balance exploration with stability

    # Calculate intrinsic reward
    if realized_volatility.mean() > risk_threshold or downside_risk.mean() > risk_threshold:
        # High-risk phase: penalize high-risk states
        intrinsic_reward_value = -alpha * (np.mean(realized_volatility) + np.mean(downside_risk) - risk_threshold)
    else:
        # Favorable conditions: reward exploration through trends
        intrinsic_reward_value = np.mean(rel_momentum) * (1 - 0.5 * risk_level)

    return intrinsic_reward_value