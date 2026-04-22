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
    # Extract close prices and volumes from the state representation
    close_prices = s[0::6]  # Extract close prices for 20 days
    volumes = s[4::6]       # Extract volumes for 20 days
    
    # Calculate daily returns
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Normalize daily returns
    
    # Calculate additional features using the provided feature library functions
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(daily_returns)
    downside_risk = compute_downside_risk(daily_returns)
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Combine new features into a single array
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

    # Concatenate original state with new features to create an updated state representation
    updated_s = np.concatenate((s, extra_features))

    return updated_s


def intrinsic_reward(updated_s):
    # Extract key features from the updated state
    relative_momentum = updated_s[120]    # Relative momentum
    realized_volatility = updated_s[121]   # Realized volatility 
    downside_risk = updated_s[122]         # Downside risk
    
    # Dynamically adjust risk_level based on external market conditions; setting a constant for this example
    risk_level = 0.33  
    
    # Define thresholds for penalizing excessive risk
    VOLATILITY_THRESHOLD = 0.05  
    DOWNSIDE_RISK_THRESHOLD = 0.03  
    
    # Calculate the intrinsic reward based on the market regime
    intrinsic_r = 0.0
    
    if risk_level > 0.5:  # In high-risk regimes
        # Penalize for high realized volatility and downside risk
        penalties = (np.maximum(realized_volatility - VOLATILITY_THRESHOLD, 0) +
                     np.maximum(downside_risk - DOWNSIDE_RISK_THRESHOLD, 0))
        intrinsic_r = -penalties  # Negative penalty to reduce risk exposure
    else:  # In favorable or balanced market conditions
        # Encourage exploration of positive relative momentum adjusted for risk level
        intrinsic_r = relative_momentum * (1 - 0.5 * risk_level)  # Adjust momentum based on current risk level

    return intrinsic_r