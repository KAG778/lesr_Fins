import numpy as np
from feature_library import (compute_relative_momentum, compute_realized_volatility,
                             compute_downside_risk, compute_multi_horizon_momentum,
                             compute_zscore_price, compute_mean_reversion_signal,
                             compute_turnover_ratio)

def revise_state(s):
    # Extract close prices and volumes for each stock over the 20-day period
    close_prices = s[0::6]
    volumes = s[4::6]
    
    # Initialize list to accumulate additional features
    extra_features = []
    
    # Calculate the required features for each stock
    for i in range(5):  # We have 5 stocks
        stock_close_prices = close_prices[i * 20:(i + 1) * 20]
        stock_volumes = volumes[i * 20:(i + 1) * 20]
        
        # Computed features
        relative_momentum = compute_relative_momentum(stock_close_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_close_prices))
        downside_risk = compute_downside_risk(np.diff(stock_close_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_close_prices)
        zscore_price = compute_zscore_price(stock_close_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_close_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)
        
        # Append features to extra_features list
        extra_features.extend([relative_momentum, realized_volatility, downside_risk] +
                               list(multi_horizon_momentum) + 
                               [zscore_price, mean_reversion_signal, turnover_ratio])
    
    # Convert the list of additional features to a NumPy array
    extra_features = np.array(extra_features)
    
    # Return the updated state by concatenating original state and extra features
    updated_s = np.concatenate((s, extra_features))
    return updated_s

def intrinsic_reward(updated_s):
    # Using some relevant sources from updated_s and extra dimensions
    # Assuming `risk_level` is available from the market strategy guidance

    # Extract the calculated features from the updated state
    # Adjusting indices according to features we added
    realized_volatility = updated_s[120:125][2]  # Downside risk for one stock (the third stock as example)
    downside_risk = updated_s[120:125][1]  # Another stock's downside risk
    
    # Using a simple strategy for intrinsic reward
    # Assume we retrieve risk information from somewhere
    risk_level = 0.07  # provided from market strategy guidance
    threshold = 0.02   # hypothetical threshold for downside risk
    
    # Decision based on market regime and risk level
    if risk_level > 0.05:  # Example threshold for high risk regime
        reward = -1 * max(0, downside_risk - threshold)
    else:
        # In favorable conditions (e.g., balanced), we would reward exploration
        reward = (1 - 0.5 * risk_level) * np.mean(updated_s[120:125])  # average of extra features
    
    return reward