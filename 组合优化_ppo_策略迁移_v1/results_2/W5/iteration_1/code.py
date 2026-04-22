import numpy as np
from feature_library import (
    compute_relative_momentum,
    compute_realized_volatility,
    compute_downside_risk,
    compute_multi_horizon_momentum,
    compute_zscore_price,
    compute_mean_reversion_signal,
    compute_turnover_ratio,
)

def revise_state(s):
    # Extract prices and volumes from the state representation
    close_prices = s[0::6]
    volumes = s[4::6]

    # Compute various additional features
    returns = np.diff(close_prices) / close_prices[:-1]  # Daily returns calculation

    # Relative Momentum
    momentum = compute_relative_momentum(close_prices)

    # Realized Volatility
    realized_vol = compute_realized_volatility(returns)

    # Downside Risk
    downside_risk = compute_downside_risk(returns)

    # Multi-horizon Momentum
    multi_mom = compute_multi_horizon_momentum(close_prices)
    
    # Z-score Price
    z_score = compute_zscore_price(close_prices)

    # Mean Reversion Signal
    mean_reversion = compute_mean_reversion_signal(close_prices)

    # Turnover Ratio
    turnover_ratio = compute_turnover_ratio(volumes)

    # Construct the updated state
    updated_s = np.concatenate(
        [s, 
         np.array([momentum, realized_vol, downside_risk] + multi_mom.tolist() + 
                   [z_score, mean_reversion, turnover_ratio])]
    )
    return updated_s

def intrinsic_reward(updated_s):
    # Extract new features and source dimensions
    risk_feature = updated_s[120:123]  # Assumed risk features are from index 120 to 122
    realized_volatility = updated_s[120]  # Realized Volatility
    downside_risk = updated_s[121]  # Downside Risk
    momentum_signal = updated_s[0]  # Using first source dimension as info signal

    # Determine the market regime (defensive as per guidance)
    defensive_market = True  # Assuming we are in a defensive market regime
    
    if defensive_market:
        # Penalize based on the risk features
        # For example, consider a threshold of median values for vol and downside risk
        threshold = 0.02  # Example threshold value
        
        # Penalty for high volatility or downside risk
        penalty = -1.0 * max(0, realized_volatility - threshold) - max(0, downside_risk - threshold)
        reward = penalty
    else:
        # Encourage exploration in a favorable regime
        reward = momentum_signal  # A simple way to reward based on momentum signal during favorable regimes

    return reward