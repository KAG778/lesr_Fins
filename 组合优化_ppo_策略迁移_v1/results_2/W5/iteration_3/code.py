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
    close_prices = s[0::6]  # Close prices from the state
    volumes = s[4::6]       # Volume data from the state

    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1]

    # Compute additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    z_score = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Construct the updated state
    updated_s = np.concatenate([
        s, 
        np.array([
            relative_momentum,
            realized_volatility,
            downside_risk,
            *multi_horizon_momentum,
            z_score,
            mean_reversion_signal,
            turnover_ratio
        ])
    ])
    return updated_s

def intrinsic_reward(updated_s):
    # Extract new features
    realized_volatility = updated_s[120]  # Realized Volatility
    downside_risk = updated_s[121]         # Downside Risk
    relative_momentum = updated_s[126]     # Relative Momentum from calculated features
    z_score = updated_s[124]                # Z-score for additional check on prices

    # Determine if we are in a defensive market regime
    market_regime = 'Defensive'  # Assumed defensive regime

    # Set thresholds for risk penalization
    threshold_volatility = 0.03  # Example threshold for realized volatility
    threshold_downside = 0.03     # Example threshold for downside risk

    if market_regime == 'Defensive':
        # Penalization for elevated risks in a defensive market
        risk_penalty = max(0, realized_volatility - threshold_volatility) + max(0, downside_risk - threshold_downside)
        reward = -1 * risk_penalty  # Negative reward for higher risks
        
        # Include a logic to reward for knowing price dynamics through z-score
        if z_score < 0:  # Indicating potential mean-reversion
            reward += abs(z_score) * 10  # Scaling the mean-reversion signal positively
    
    else:
        # Reward based on positive relative momentum in favorable regimes
        reward = relative_momentum * 100  # Scale momentum for better feedback

    return reward