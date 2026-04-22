import numpy as np
from feature_library import (compute_relative_momentum, compute_realized_volatility,
                              compute_downside_risk, compute_multi_horizon_momentum,
                              compute_zscore_price, compute_mean_reversion_signal,
                              compute_turnover_ratio)

def revise_state(s):
    # Reshape the state to separate the 20 days of features for each stock
    num_days = 20
    num_assets = 6  # Close, Open, High, Low, Volume, Adjusted Close
    close_prices = s[0::6]
    volumes = s[4::6]

    # Initialize the additional features
    additional_features = []

    for stock_idx in range(5):  # for TSLA, NFLX, AMZN, MSFT, JNJ
        stock_prices = close_prices[stock_idx:num_assets*num_days:num_assets]
        stock_volumes = volumes[stock_idx:num_assets*num_days:num_assets]

        # Compute features
        momentum = compute_relative_momentum(stock_prices)  # Excess return vs window-average
        realized_vol = compute_realized_volatility(np.diff(stock_prices))  # Daily returns
        downside_risk = compute_downside_risk(np.diff(stock_prices))  # Daily returns
        multi_horizon_mom = compute_multi_horizon_momentum(stock_prices)  # Trend at multiple time scales
        zscore = compute_zscore_price(stock_prices)  # Z-score of current price
        mean_reversion = compute_mean_reversion_signal(stock_prices)  # Mean reversion strength
        turnover = compute_turnover_ratio(stock_volumes)  # Current volume / average volume

        # Append features to the additional_features list
        additional_features.extend([momentum, realized_vol, downside_risk] + list(multi_horizon_mom) +
                                   [zscore, mean_reversion, turnover])

    # Convert additional_features to Numpy array
    additional_features = np.array(additional_features)

    # Create the updated state
    updated_s = np.concatenate((s, additional_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features for the reward calculation
    realized_volatility = updated_s[120:125]  # Assuming these contain realized volatilities
    downside_risk = updated_s[125:130]  # Assuming these contain downside risks
    momentum_features = updated_s[130:135]  # Assuming these contain momentum features

    # Market regime considerations (feature selection guidance mentioned)
    risk_level = 0.33  # placeholder for actual risk level, would be from environment
    alpha = 1.0  # penalty scaling factor

    # Determine reward based on market conditions
    if risk_level < 0.5:  # Favorable market conditions
        reward = np.mean(momentum_features)  # Encourage exploration based on momentum
    else:  # Risky market conditions (defensive/crisis)
        # Penalize based on maximum realized volatility or downside risk
        penalty = alpha * max(0, np.mean(realized_volatility) - 0.2)  # Threshold can be adjusted
        reward = -penalty

    return reward