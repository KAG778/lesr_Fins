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
    # Reshape state into structured data for each stock
    num_assets = 5  # TSLA, NFLX, AMZN, MSFT, JNJ
    num_days = 20
    channels = 6
    close_prices = s[0::6]  # close prices
    volumes = s[4::6]  # volumes
    
    additional_features = []

    # Gather features for each stock over the 20-day period
    for stock_idx in range(num_assets):
        stock_prices = close_prices[stock_idx:num_assets*num_days:num_assets]
        stock_volumes = volumes[stock_idx:num_assets*num_days:num_assets]     
        
        # Calculate various features for the stock
        momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(np.diff(stock_prices))
        downside_risk = compute_downside_risk(np.diff(stock_prices))
        multi_horizon_mom = compute_multi_horizon_momentum(stock_prices)
        z_score_price = compute_zscore_price(stock_prices)
        mean_reversion = compute_mean_reversion_signal(stock_prices)
        turnover = compute_turnover_ratio(stock_volumes)

        # Append features to the additional features list
        additional_features.extend([
            momentum,       
            realized_volatility,
            downside_risk,
            *multi_horizon_mom,  # Array of form [momentum_5, momentum_10, momentum_20]
            z_score_price,
            mean_reversion,
            turnover
        ])

    updated_s = np.concatenate((s, additional_features))

    return updated_s

def intrinsic_reward(updated_s):
    # Extract features for reward computation (feature selection guided by market conditions)
    realized_volatility = updated_s[120:125]  # Assuming these are the realized volatilities
    downside_risk = updated_s[125:130]  # Assuming these are the downside risks
    momentum_features = updated_s[130:135]  # Assuming these are the momentum features

    # Example risk level - this would typically come from your environment
    risk_level = 0.33  # Placeholder, example for current risk condition

    # Define a base reward
    reward = np.mean(momentum_features)  # Base reward based on momentum

    # In high-risk conditions, penalize the reward based on realized volatility or downside risk
    if risk_level >= 0.5:
        penalty_realized_vol = np.mean(realized_volatility) - 0.2  # Adjust this threshold as necessary
        penalty_downside_risk = np.mean(downside_risk) - 0.1  # Adjust threshold
        penalty = max(0, penalty_realized_vol) + max(0, penalty_downside_risk)  # If below thresholds, no penalty
        reward -= penalty  # Penalize reward for risk exposure

    return reward