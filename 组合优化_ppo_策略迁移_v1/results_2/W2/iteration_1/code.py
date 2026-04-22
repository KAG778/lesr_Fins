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
    # Extract close prices and volumes for calculation
    close_prices = s[0::6]  # Close prices (20 values)
    volumes = s[4::6]       # Volumes (20 values)
    
    # Compute various features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(np.diff(close_prices))
    downside_risk = compute_downside_risk(np.diff(close_prices))
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    zscore_current_price = compute_zscore_price(close_prices)
    mean_reversion_strength = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Combine the original state with computed features
    updated_s = np.concatenate((s, np.array([
        relative_momentum,
        realized_volatility,
        downside_risk,
        *multi_horizon_momentum,
        zscore_current_price,
        mean_reversion_strength,
        turnover_ratio
    ])))

    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant features from updated state
    realized_volatility = updated_s[120]  # Updated dimension for realized volatility
    downside_risk = updated_s[121]         # Updated dimension for downside risk

    # Market regime guidance assessment (Assuming we signify a balanced market here)
    is_favorable_regime = True  # Change depending on actual regime detection logic
    risk_threshold = 0.03       # Example threshold for penalties on risk
    
    if is_favorable_regime:
        # In favorable regime, reward for low realized volatility and downside risk
        reward = -0.5 * realized_volatility - 0.5 * downside_risk
    else:
        # In high risk regime, penalize high realized volatility or downside risk
        reward = -1.0 * max(0, realized_volatility - risk_threshold) - 1.0 * max(0, downside_risk - risk_threshold)

    return reward

# Example usage (will only work if feature_library is properly defined)
# s = np.random.random(120)  # Example input state
# updated_s = revise_state(s)
# reward = intrinsic_reward(updated_s)