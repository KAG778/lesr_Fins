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
    num_assets = 5  # 5 stocks
    num_days = 20   # Lookback period of 20 days
    dimensions_per_stock = 6  # Slicing size for each stock

    close_prices = s[0::6]  # Close prices for all days of all stocks
    volumes = s[4::6]       # Volume data

    # Initialize lists to hold computed features
    extra_features = []

    # Compute features for each stock
    for i in range(num_assets):
        idx = i * num_days
        prices = close_prices[idx: idx + num_days]  # Get the last 20 days of close prices for the stock
        current_volumes = volumes[idx: idx + num_days]  # Get the volumes

        # Calculate feature metrics for the current stock
        rel_momentum = compute_relative_momentum(prices)
        realized_volatility = compute_realized_volatility(np.diff(prices) / prices[:-1])
        downside_risk = compute_downside_risk(np.diff(prices) / prices[:-1])
        multi_horizon_mom = compute_multi_horizon_momentum(prices)
        zscore_price = compute_zscore_price(prices)
        mean_reversion_signal = compute_mean_reversion_signal(prices)
        turnover_ratio = compute_turnover_ratio(current_volumes)

        # Combine features for this stock into the list
        extra_features.extend([rel_momentum, realized_volatility, downside_risk] +
                              list(multi_horizon_mom) + [zscore_price, mean_reversion_signal, turnover_ratio])

    # Convert the extra_features list to a numpy array and append to the original state
    extra_features = np.array(extra_features)
    updated_s = np.concatenate((s, extra_features))

    return updated_s


def intrinsic_reward(updated_s):
    num_assets = 5  # Number of stocks
    realized_volatility = updated_s[120:120 + num_assets]  # Realized volatility for each stock
    downside_risk = updated_s[125:125 + num_assets]        # Downside risk for each stock
    turnover_ratio = updated_s[130:130 + num_assets]       # Turnover ratio for each stock

    # Example market regime (to be replaced with actual regime detection)
    market_regime = 'Balanced'  # Could be 'Defensive', 'Crisis', etc.

    if market_regime in ['Defensive', 'Crisis']:
        # In high-risk regimes, penalize for high volatility and downside risk
        volatility_threshold = 0.04  # 4% benchmark
        penalty_volatility = max(0, np.mean(realized_volatility) - volatility_threshold)
        penalty_downside = np.mean(downside_risk)
        reward = -1 * (penalty_volatility + penalty_downside)  # Total penalty
    else:
        # In favorable regimes, reward exploration of informative features
        reward = np.mean(realized_volatility) - 0.5 * np.mean(downside_risk) + np.mean(turnover_ratio)

    return reward