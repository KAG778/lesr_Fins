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
    # Number of days (20) and the number of stocks (excluding cash)
    num_days = 20
    num_stocks = 5  # TSLA, NFLX, AMZN, MSFT, JNJ

    # Initialize a list to hold the calculated features
    extra_features = []

    # Reshape the state `s` to extract price and volume information
    close_prices = s[0::6]  # Close prices for 20 days
    volumes = s[4::6]       # Volume data for 20 days

    for i in range(num_stocks):  # Loop through each stock
        stock_prices = close_prices[i * num_days:(i + 1) * num_days]
        stock_volumes = volumes[i * num_days:(i + 1) * num_days]

        # Calculate features
        momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_prices))  # Daily returns
        downside_risk = compute_downside_risk(np.diff(stock_prices))  # Daily returns
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_prices)
        zscore = compute_zscore_price(stock_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)

        # Append calculated features to the extra_features list
        extra_features.extend([
            momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            zscore,
            mean_reversion_signal,
            turnover_ratio
        ])
    
    # Combine the original state with the additional features
    updated_s = np.concatenate((s, extra_features))
    
    return updated_s


def intrinsic_reward(updated_s):
    # Extract relevant features from the updated state
    realized_volatility = updated_s[120:125]  # Realized volatility for each of the 5 stocks
    downside_risk = updated_s[125:130]         # Downside risk for each of the 5 stocks
    trend_signal = updated_s[120:125:3]        # Momentum for the 5 stocks

    # Example market regime assessment (you may replace this logic with your own)
    market_regime = "favorable"  # Placeholder (this should use actual market regime detection)
    risk_threshold = 0.03  # Example threshold for penalizing high-risk states
    
    # Calculate the reward based on the market regime
    if market_regime == "favorable":
        # Reward for healthy momentum while reducing impact of realized volatility
        reward = np.mean(trend_signal) * (1 - 0.5 * np.mean(realized_volatility))
    else:
        # Penalize for high volatility and downside risk in unfavorable regimes
        penalty_realized_volatility = np.maximum(0, np.max(realized_volatility) - risk_threshold)
        penalty_downside_risk = np.maximum(0, np.max(downside_risk) - risk_threshold)
        reward = -1.0 * (penalty_realized_volatility + penalty_downside_risk)
    
    return reward

# Example usage:
# s = np.random.random(120)  # Replace this with actual state data
# updated_s = revise_state(s)
# reward = intrinsic_reward(updated_s)