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
    # Initialize arrays to hold features for each stock
    num_stocks = 5
    momentum_features = np.zeros(num_stocks)  # Relative momentum for each stock
    volatility_features = np.zeros(num_stocks)  # Realized volatility for each stock
    downside_risk_features = np.zeros(num_stocks)  # Downside risk for each stock
    multi_horizon_momentum_features = np.zeros((num_stocks, 3))  # 3 horizons for each stock
    mean_reversion_features = np.zeros(num_stocks)  # Mean reversion signal for each stock
    turnover_features = np.zeros(num_stocks)  # Turnover ratio for each stock

    # Loop through each stock to compute the features
    for i in range(num_stocks):
        # Extract relevant prices and volumes for each stock
        close_prices = s[i * 6:i * 6 + 120:6]  # Close prices for 20 days
        volumes = s[i * 6 + 4:i * 6 + 120 + 4:6]  # Volume for 20 days

        # Calculate the features
        momentum_features[i] = compute_relative_momentum(close_prices)
        returns = np.diff(close_prices) / close_prices[:-1]  # Daily returns
        volatility_features[i] = compute_realized_volatility(returns)
        downside_risk_features[i] = compute_downside_risk(returns)
        multi_horizon_momentum_features[i] = compute_multi_horizon_momentum(close_prices)
        mean_reversion_features[i] = compute_mean_reversion_signal(close_prices)
        turnover_features[i] = compute_turnover_ratio(volumes)

    # Concatenate new features to the original state
    updated_s = np.concatenate((
        s,
        momentum_features,
        volatility_features,
        downside_risk_features,
        multi_horizon_momentum_features.flatten(),
        mean_reversion_features,
        turnover_features
    ))
    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features from the updated state
    relative_momentum = updated_s[120:125]  # Relative momentum for each stock
    realized_volatility = updated_s[125:130]  # Realized volatility for each stock
    downside_risk = updated_s[130:135]  # Downside risk for each stock
    trend_momentum_5d = updated_s[135:138]  # Multi-horizon momentum (5 days)

    # Market regime guidance
    risk_level = 0.14  # Provided risk level in the guidance
    risk_threshold = 0.03  # Threshold for high risk
    alpha = 0.01  # Factor for risk adjustment

    # Compute intrinsic reward based on the market regime
    intrinsic_r = 0  # Initialize intrinsic reward
    if realized_volatility.mean() > risk_threshold or downside_risk.mean() > risk_threshold:
        # High-risk regime: penalize high risk states
        penalty = np.sum(realized_volatility[realized_volatility > risk_threshold]) + np.sum(downside_risk)
        intrinsic_r = -alpha * penalty  # Penalize the total risk
    else:
        # Favorable conditions: encourage exploration through trends
        intrinsic_r = np.mean(relative_momentum) * (1 - 0.5 * risk_level)  # Balance exploration with risk

    return intrinsic_r