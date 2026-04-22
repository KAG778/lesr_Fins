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
    price_channels = 6  # [close, open, high, low, volume, adjusted_close]
    
    # Reshaping the state
    close_prices = np.array(s[0::6])  # Extracting 20 days of close prices
    volumes = np.array(s[4::6])       # Extracting 20 days of volumes

    extra_features = []
    
    for i in range(num_stocks):
        stock_prices = close_prices[i*days:(i+1)*days]
        stock_volumes = volumes[i*days:(i+1)*days]

        # Calculate daily returns
        returns = np.diff(stock_prices) / stock_prices[:-1]

        # Compute additional features
        relative_momentum = compute_relative_momentum(stock_prices)
        realized_volatility = compute_realized_volatility(returns)
        downside_risk = compute_downside_risk(returns)
        multi_horizon_momentum = compute_multi_horizon_momentum(stock_prices)
        zscore_price = compute_zscore_price(stock_prices)
        mean_reversion_signal = compute_mean_reversion_signal(stock_prices)
        turnover_ratio = compute_turnover_ratio(stock_volumes)

        # Gather all calculated features
        extra_features.extend([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            zscore_price,
            mean_reversion_signal,
            turnover_ratio
        ])
    
    # Combine original state with the new features
    updated_s = np.concatenate((s, extra_features))

    return updated_s

def intrinsic_reward(updated_s):
    # Extracting features for calculating intrinsic reward
    realized_volatility = updated_s[120]  # First additional feature
    downside_risk = updated_s[121]         # Second additional feature

    # Define market regime based on external analysis, e.g., from a model or signal
    market_regime = "Defensive"  # Placeholder - this should come from an external signal
    epsilon = 1e-10  # Small constant to avoid division by zero
    alpha = 2.0  # Increased penalty weight for risk factors for better sensitivity

    if market_regime == "Defensive":
        # Penalizing high realized volatility and downside risk
        risk_penalty = alpha * (realized_volatility + downside_risk - 0.05)  # Threshold of 0.05
        intrinsic_reward_value = -max(0, risk_penalty)  # Inverse reward
    else:
        # Encouraging states in a favorable regime by averaging the first 120 features (original state info)
        intrinsic_reward_value = np.mean(updated_s[0:120])  # Exploratory encouragement
    
    return intrinsic_reward_value