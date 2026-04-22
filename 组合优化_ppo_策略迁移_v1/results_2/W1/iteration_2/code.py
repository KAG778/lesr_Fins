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
    num_assets = 5  # Number of assets excluding CASH
    features_per_day = 6  # Close, open, high, low, volume, adjusted_close

    # Reshape the state representation to extract the relevant prices and volumes
    prices = s[0::features_per_day]  # Extracting close prices
    volumes = s[4::features_per_day]  # Extracting volume information
    
    # Calculate required features
    returns = np.diff(prices) / prices[:-1]  # Calculate daily returns
    relative_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(prices)
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Create the updated state representation by combining original and newly computed features
    updated_s = np.concatenate([
        s, 
        np.array([
            relative_momentum,       # 1 scalar
            realized_volatility,     # 1 scalar
            downside_risk,          # 1 scalar
            *multi_horizon_momentum, # 3 scalars for 5, 10, 20 days momentum
            zscore_price,           # 1 scalar
            mean_reversion_signal,   # 1 scalar
            turnover_ratio           # 1 scalar
        ])
    ])
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features from the updated state representation
    relative_momentum = updated_s[120]   # Scalar: relative momentum
    realized_volatility = updated_s[121]  # Scalar: realized volatility
    downside_risk = updated_s[122]        # Scalar: downside risk

    # Define the market regime parameters based on current conditions
    market_regime = "Balanced"  # This should be dynamically determined in the environment
    risk_level = 0.15  # Risk level can also vary based on market conditions
    
    # Calculate the reward based on market conditions
    if market_regime in ["Aggressive", "Balanced"]:
        # Encouraging exploration in favorable market conditions
        reward = relative_momentum - 0.5 * realized_volatility
    else:
        # Applying penalties in riskier market scenarios
        penalty = max(0, realized_volatility - 0.04)  # Example penalty threshold
        reward = relative_momentum - penalty - 0.5 * downside_risk

    return reward