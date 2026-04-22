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
    
    # Reshape the state representation for easier processing
    prices = s[0::6]
    volumes = s[4::6]
    
    # Compute additional features for each stock
    updated_features = []
    
    for i in range(num_stocks):
        stock_prices = prices[i * days:(i + 1) * days]
        stock_volumes = volumes[i * days:(i + 1) * days]
        
        # Compute additional features
        relative_momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_prices))  # Daily returns
        downside_risk = compute_downside_risk(np.diff(stock_prices))
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_prices)
        zscore_price = compute_zscore_price(stock_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)
        
        # Append computed features to the updated features list
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
    
    # Convert updated features list to a NumPy array
    updated_features = np.array(updated_features)

    # Concatenate the original state with the updated features
    updated_s = np.concatenate((s, updated_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract original features from the updated state
    prices = updated_s[0::6]
    realized_volatility = updated_s[120:125]  # Assuming indexes [120:125] correspond to the realized volatility of each stock
    downside_risk = updated_s[125:130]  # Assuming indexes [125:130] correspond to the downside risk of each stock
    relative_momentum = updated_s[130:135]  # Assuming indexes [130:135] correspond to the relative momentum
    
    # Define thresholds and constants
    volatility_threshold = 0.03  # Example threshold for realized volatility
    alpha = 1.0  # Scaling factor for penalty or incentive
    
    # Check the current market regime (for demonstration, we assume it's Balanced here)
    market_regime = "Balanced"  # Would be dynamic in a real implementation
    
    if market_regime == "Balanced":
        # Reward exploration by promoting states with informative features
        reward = np.mean(relative_momentum)  # Simple mean of relative momentums
    else:
        # If the regime indicates high risk, penalize high volatility or downside risk
        reward = -alpha * np.sum(np.maximum(0, realized_volatility - volatility_threshold))
        
    return reward