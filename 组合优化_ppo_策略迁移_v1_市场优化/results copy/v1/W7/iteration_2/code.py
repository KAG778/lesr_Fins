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
    # Initialize the updated state list
    updated_s = list(s)

    # Extract relevant features from the state for calculations
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Volumes

    # Calculate daily returns for further analysis
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Compute various metrics to append to the state vectors
    relative_momentum = compute_relative_momentum(closing_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(closing_prices)
    zscore_price = compute_zscore_price(closing_prices)
    mean_reversion_signal = compute_mean_reversion_signal(closing_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Append computed features to the original state
    updated_s.extend([
        relative_momentum,        # Relative momentum
        realized_volatility,      # Realized volatility
        downside_risk,           # Downside risk
        *multi_horizon_momentum,  # Multi-horizon momentums: [5d, 10d, 20d]
        zscore_price,            # Z-score of price
        mean_reversion_signal,    # Mean reversion signal
        turnover_ratio            # Turnover ratio
    ])

    # Convert back to a numpy array
    return np.array(updated_s)

def intrinsic_reward(updated_s):
    # Extract features relevant for the reward calculation
    relative_momentum = updated_s[120]    # Assume this is the index for relative momentum
    realized_volatility = updated_s[121]   # Assume this is the index for realized volatility
    downside_risk = updated_s[122]         # Assume this is the index for downside risk

    # Market regime settings (Adjust according to dynamic conditions if needed)
    risk_level = 0.14  # Example risk level representing mild market uncertainty
    
    # Set thresholds for distinguishing risk
    volatility_threshold = 0.1      # Threshold for volatility
    downside_risk_threshold = 0.05   # Threshold for downside risk

    # Calculate intrinsic reward based on market conditions
    if risk_level > 0.1:
        # High-risk regime: Penalties for high volatility and downside risk
        penalty = (
            -1 * max(0, realized_volatility - volatility_threshold) +
            -1 * max(0, downside_risk - downside_risk_threshold)
        )
        reward = penalty
    else:
        # Favorable regime: Reward exploration of performance characteristics
        reward = relative_momentum * (1 - 0.5 * realized_volatility)  # Exploration incentivized

    return reward