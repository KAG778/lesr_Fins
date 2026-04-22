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
    # Reshape the input state for easier access to features
    close_prices = s[0::6]  # Close prices for 20 days
    volumes = s[4::6]       # Volumes for 20 days
    
    # Calculate daily returns
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Normalize returns
    
    # Calculate additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Compile additional features together
    extra_features = np.array([
        relative_momentum,
        realized_volatility,
        downside_risk,
        multi_horizon_mom[0],  # 5-day momentum
        multi_horizon_mom[1],  # 10-day momentum
        multi_horizon_mom[2],  # 20-day momentum
        zscore_price,
        mean_reversion_signal,
        turnover_ratio
    ])
    
    # Concatenate the original state with all the new features
    updated_s = np.concatenate((s, extra_features))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract important features from the updated state
    relative_momentum = updated_s[120]  # Relative momentum
    realized_volatility = updated_s[121]  # Realized volatility
    downside_risk = updated_s[122]  # Downside risk
    risk_level = 0.33  # Assuming the current risk level (Example: balanced market)
    
    # Define reward thresholds
    VOLATILITY_THRESHOLD = 0.05  # Adjust based on analysis
    DOWNSIDE_RISK_THRESHOLD = 0.03  # Adjust based on analysis
    
    # Calculate intrinsic reward based on current model guidance
    if risk_level > 0.5:  # High-risk regime
        # Penalize if volatility or downside risk exceeds the threshold
        penalty = 0.0
        if realized_volatility > VOLATILITY_THRESHOLD:
            penalty -= (realized_volatility - VOLATILITY_THRESHOLD)
        if downside_risk > DOWNSIDE_RISK_THRESHOLD:
            penalty -= (downside_risk - DOWNSIDE_RISK_THRESHOLD)
        
        intrinsic_r = penalty  # Focus on reducing risk
    else:  # Balanced or aggressive regime
        # Encourage features that drive momentum in portfolios
        intrinsic_r = relative_momentum  # Reward based on momentum
    
    return intrinsic_r