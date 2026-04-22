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
    # Initialize lists for computed features
    num_stocks = 5
    additional_features = []

    # Loop through each stock to extract and compute features
    for i in range(num_stocks):
        # Extract the close prices and volumes for the current stock
        prices = s[i * 6: (i + 1) * 6][0::6]
        volumes = s[i * 6: (i + 1) * 6][4::6]

        # Compute additional features
        rel_momentum = compute_relative_momentum(prices)
        realized_volatility = compute_realized_volatility(np.diff(prices) / prices[:-1])
        downside_risk = compute_downside_risk(np.diff(prices) / prices[:-1])
        multi_horizon_momentum = compute_multi_horizon_momentum(prices)
        zscore_price = compute_zscore_price(prices)
        mean_reversion_signal = compute_mean_reversion_signal(prices)
        turnover_ratio = compute_turnover_ratio(volumes)

        # Collect additional features
        additional_features.extend([
            rel_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            zscore_price,
            mean_reversion_signal,
            turnover_ratio
        ])

    # Convert additional features to numpy array
    additional_features = np.array(additional_features)

    # Concatenate additional features with the original state
    updated_state = np.concatenate((s, additional_features))
    return updated_state

def intrinsic_reward(updated_s):
    # Extract relevant features
    rel_momentum = updated_s[120]  # relative momentum
    realized_volatility = updated_s[121]  # realized volatility
    downside_risk = updated_s[122]  # downside risk
    trend_momentum_5d = updated_s[123]  # multi-horizon momentum (5 days)
    risk_level = 0.14  # Given risk level from the market guidance

    # Define intrinsic reward based on the market regime
    alpha = 0.01  # Coefficient to balance exploration with stability
    risk_threshold = 0.03  # Example threshold for risk

    # Determine the reward based on the risk condition
    if realized_volatility > risk_threshold or downside_risk > risk_threshold:
        # High-risk phase: penalize high risk states
        intrinsic_r = -alpha * (realized_volatility + downside_risk - risk_threshold)
    else:
        # Favorable conditions: encourage exploration through trends
        intrinsic_r = rel_momentum * (1 - 0.5 * risk_level)

    return intrinsic_r