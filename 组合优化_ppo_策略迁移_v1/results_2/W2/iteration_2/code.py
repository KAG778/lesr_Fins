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
    # Number of days and stocks (excluding cash)
    num_days = 20
    num_stocks = 5  # 5 stocks only (TSLA, NFLX, AMZN, MSFT, JNJ)
    
    # Initialize lists to hold calculated features
    extra_features = []
    
    # Reshape the state s to extract information for each stock
    prices = s[0::6]  # Extracting close prices
    volumes = s[4::6]  # Extracting volume data
    
    for i in range(num_stocks):  # Loop through each stock
        stock_prices = prices[i * num_days:(i + 1) * num_days]
        stock_volumes = volumes[i * num_days:(i + 1) * num_days]
        
        # Calculate various features
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
            downside_risk
        ] + multi_horizon_momentum.tolist() + [
            zscore, 
            mean_reversion_signal, 
            turnover_ratio
        ])
    
    # Combine original state with extra features
    updated_s = np.concatenate((s, extra_features))
    return updated_s

def intrinsic_reward(updated_s):
    # Extract features from updated state
    stock_realized_volatility = updated_s[120:120 + 5]  # Realized volatility for each of the 5 stocks
    stock_downside_risk = updated_s[120 + 5:120 + 10]  # Downside risk for each of the 5 stocks
    stock_momentum = updated_s[120 + 10:120 + 14]  # Momentum for the 5 stocks
    
    # Current market regime assessment
    is_favorable_regime = True  # Change this to reflect your market regime detection logic
    risk_threshold = 0.03  # Threshold for penalizing high volatilities
    
    if is_favorable_regime:
        # Encourage exploration of informative features in favorable regimes
        reward = np.mean(stock_momentum) * (1 - 0.5 * np.mean(stock_realized_volatility))  # Reward trend-adjusted by risk
    else:
        # Penalize high risk in unfavorable regimes
        max_volatility = np.max(stock_realized_volatility)
        max_downside = np.max(stock_downside_risk)
        reward = -1.0 * (max(0, max_volatility - risk_threshold) + max(0, max_downside - risk_threshold))  # Penalties
    
    return reward