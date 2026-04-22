import numpy as np
from feature_library import (compute_relative_momentum, compute_realized_volatility,
                             compute_downside_risk, compute_multi_horizon_momentum,
                             compute_zscore_price, compute_mean_reversion_signal,
                             compute_turnover_ratio)

def revise_state(s):
    # Extract close prices and volumes for each stock over the 20-day period
    close_prices = s[0::6]
    volumes = s[4::6]
    
    # Initialize a list to accumulate additional features
    additional_features = []
    
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
        
        # Append features to the list
        additional_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,  # Expand tuple to list
            zscore_price,
            mean_reversion_signal,
            turnover_ratio
        ])

    # Convert additional_features to a NumPy array
    additional_features = np.array(additional_features)

    # Return the updated state by concatenating the original state and extra features
    updated_s = np.concatenate((s, additional_features))
    return updated_s

def intrinsic_reward(updated_s):
    # Assume fixed risk level from market strategy guidance
    risk_level = 0.07  # Current risk level as per market regime guidance
    downside_risk_threshold = 0.02  # Threshold for downside risk

    # Extract necessary metrics
    realized_volatilities = updated_s[120:125]  # Get realized volatilities for stocks
    downside_risks = updated_s[125:130]  # Get downside risks for stocks
    mean_momentum = np.mean(updated_s[120:165])  # Mean of trend features included in updated_state

    # Calculate average risks
    avg_downside_risk = np.mean(downside_risks)
    avg_realized_volatility = np.mean(realized_volatilities)

    # Decision logic for intrinsic reward calculation
    if risk_level > 0.05:  # High risk regime
        # Penalizing high downside risk in high-risk environments
        reward = -np.sum(np.maximum(0, downside_risks - downside_risk_threshold))
    else:
        # In balanced or favorable environments, encourage exploration
        reward = mean_momentum * (1 - 0.5 * risk_level)  # Incorporating risk adjustment

    return float(reward)