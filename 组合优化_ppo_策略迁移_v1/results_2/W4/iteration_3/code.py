import numpy as np
from feature_library import (compute_relative_momentum, compute_realized_volatility,
                             compute_downside_risk, compute_multi_horizon_momentum,
                             compute_zscore_price, compute_mean_reversion_signal,
                             compute_turnover_ratio)

def revise_state(s):
    # Extract close prices and volumes for all stocks over the 20-day period
    close_prices = s[0::6]
    volumes = s[4::6]

    # Initialize a list to store additional features
    additional_features = []
    
    # Calculate the required features for each stock
    for i in range(5):  # We have 5 stocks
        stock_close_prices = close_prices[i * 20:(i + 1) * 20]
        stock_volumes = volumes[i * 20:(i + 1) * 20]

        # Compute necessary features
        relative_momentum = compute_relative_momentum(stock_close_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_close_prices))
        downside_risk = compute_downside_risk(np.diff(stock_close_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_close_prices)
        zscore_price = compute_zscore_price(stock_close_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_close_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)

        # Store features in order
        additional_features.extend([
            relative_momentum, 
            realized_volatility, 
            downside_risk,
            *multi_horizon_momentum, 
            zscore_price, 
            mean_reversion_signal,
            turnover_ratio
        ])

    # Convert additional_features to a NumPy array
    additional_features = np.array(additional_features)

    # Concatenate original state with additional features
    updated_s = np.concatenate((s, additional_features))
    return updated_s


def intrinsic_reward(updated_s):
    # Assume a fixed risk level from market strategy guidance
    risk_level = 0.07  # Example from guidance
    downside_risk_threshold = 0.02  # Hypothetical threshold for downside risk

    # Extract the relevant metrics from updated state
    downside_risks = updated_s[120:125][2:5]  # Assuming downside risk for stocks 3 to 5
    realized_volatility = updated_s[120:125][1:4]  # Assuming realized volatilities for stocks 2 to 4

    avg_downside_risk = np.mean(downside_risks)
    avg_realized_volatility = np.mean(realized_volatility)

    # Decision logic for reward
    if risk_level > 0.05:  # High risk regime
        reward = -1 * max(0, avg_downside_risk - downside_risk_threshold)  # Penalize excessive downside risk
    else:
        # Favorable conditions - reward towards exploration
        trend_signal = np.mean(updated_s[120:])  # Mean metrics encouraging exploration
        reward = trend_signal * (1 - 0.5 * risk_level)  # Balancing trend and risk level adjustment

    return reward