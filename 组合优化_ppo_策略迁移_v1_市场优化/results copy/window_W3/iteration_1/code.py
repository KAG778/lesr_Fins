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
    # Extracting closing prices and volumes for computations
    prices = s[0::6]          # Close prices (20 values)
    volumes = s[4::6]         # Volume values (20 values)
    
    # Calculate returns for the last 20 days
    returns = np.diff(prices) / prices[:-1]  # Daily returns (19 values)
    
    # Feature calculations
    rel_momentum = compute_relative_momentum(prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_mom = compute_multi_horizon_momentum(prices)
    zscore_price = compute_zscore_price(prices)
    mean_reversion_signal = compute_mean_reversion_signal(prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Combine original state with new feature set
    updated_s = np.concatenate((
        s,                      # original 120 dimensions
        np.array([
            rel_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_mom,
            zscore_price,
            mean_reversion_signal,
            turnover_ratio
        ])
    ))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extracting features from the updated state
    realized_volatility = updated_s[120]              # Realized volatility
    downside_risk = updated_s[121]                    # Downside risk
    rel_momentum = updated_s[122]                     # Relative momentum
    mean_reversion_signal = updated_s[123]            # Mean reversion signal
    
    # Hyperparameters
    penalty_threshold = 0.05  # Example threshold for penalizing risk features
    aggressive_regime = False  # Assume a moderate regime as indicated

    # Rewards based on the current market regime
    if aggressive_regime:
        # Favorably reward exploration
        reward = rel_momentum + mean_reversion_signal
    else:
        # Penalizes high volatility and downside risk
        reward = -1 * max(0, realized_volatility - penalty_threshold) - max(0, downside_risk - penalty_threshold)
    
    return reward