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
    num_stocks = 5  # 5 stocks + CASH
    days = 20       # 20 days of data
    channels = 6    # 6 features per stock

    # Extract close prices and volumes
    prices = s[0::6]  # Close prices (20 values per stock)
    volumes = s[4::6]  # Volume values (20 values per stock)

    # Prepare list to collect new features
    features = []

    for stock_idx in range(num_stocks):
        stock_prices = prices[stock_idx * days:(stock_idx + 1) * days]
        stock_volumes = volumes[stock_idx * days:(stock_idx + 1) * days]

        # Calculate the daily returns
        daily_returns = np.diff(stock_prices) / stock_prices[:-1]

        # Feature computations
        relative_momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(daily_returns)
        downside_risk = compute_downside_risk(daily_returns)
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_prices)
        z_score_price = compute_zscore_price(stock_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)

        # Append calculated features
        features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            z_score_price,
            mean_reversion_signal,
            turnover_ratio
        ])

    # Combine original state with new features
    updated_s = np.concatenate((s, np.array(features)))

    return updated_s

def intrinsic_reward(updated_s):
    # Extracting the additional features from the updated state
    realized_volatility = updated_s[120:125]  # Realized volatility for 5 stocks
    downside_risk = updated_s[125:130]        # Downside risk for 5 stocks
    relative_momentum = updated_s[130:135]    # Relative momentum for 5 stocks
    mean_reversion_signal = updated_s[135:140]  # Mean reversion signal for 5 stocks

    # Define thresholds and variables
    risk_threshold = 0.05  # Example threshold for penalizing high risk
    regime = "balanced"     # Contextual market regime (This should be determined externally)

    if regime in ["aggressive", "balanced"]:
        # Favorable conditions to encourage exploration of positive signals
        reward = np.mean(relative_momentum) + np.mean(mean_reversion_signal)  # Add exploration rewards
    else:
        # For risk-averse regimes, penalize high realized volatility and downside risk
        penalty = np.sum(np.maximum(realized_volatility - risk_threshold, 0)) + \
                  np.sum(np.maximum(downside_risk - risk_threshold, 0))
        reward = -penalty  # Reward is negative due to penalty

    return reward