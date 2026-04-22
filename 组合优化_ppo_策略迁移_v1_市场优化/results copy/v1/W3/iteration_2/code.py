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
    days = 20
    num_features_per_day = 6
    
    # Extract prices and volumes
    close_prices = s[0::6]  # Close prices
    volumes = s[4::6]       # Volume data
    updated_features = []

    for i in range(num_stocks):
        stock_prices = close_prices[i * days:(i + 1) * days]
        stock_volumes = volumes[i * days:(i + 1) * days]

        # Calculate additional features for each stock
        relative_momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_prices))
        downside_risk = compute_downside_risk(np.diff(stock_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_prices)
        zscore_price = compute_zscore_price(stock_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)

        # Collect features
        updated_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            multi_horizon_momentum[0],  # 5-day momentum
            multi_horizon_momentum[1],  # 10-day momentum
            multi_horizon_momentum[2],  # 20-day momentum
            zscore_price,
            mean_reversion_signal,
            turnover_ratio
        ])

    # Convert the updated features list to a NumPy array
    updated_features = np.array(updated_features)
    # Concatenate the original state with the updated features
    updated_s = np.concatenate((s, updated_features))

    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features
    relative_momentum = updated_s[120:125]  # Assuming features start from index 120
    realized_volatility = updated_s[125:130] # Realized Volatility
    downside_risk = updated_s[130:135]       # Downside Risk

    # Define thresholds and constants
    volatility_threshold = 0.05  # Example threshold for risk
    alpha = 1.0  # Scaling factor for penalties/incentives

    # Determine the current market regime
    market_regime = "Balanced"  # This should be dynamically determined in a complete implementation

    if market_regime in ["Balanced", "Aggressive"]:
        # Reward exploration of informative features
        reward = np.mean(relative_momentum) * (1 - np.mean(realized_volatility) - np.mean(downside_risk))
    elif market_regime == "Crisis":
        # Penalize high volatility or downside risk
        penalty = np.sum(np.maximum(0, realized_volatility - volatility_threshold)) + \
                  np.sum(np.maximum(0, downside_risk - volatility_threshold))
        reward = -alpha * penalty 
    else:
        # Fallback to neutral behavior
        reward = 0
    
    return reward