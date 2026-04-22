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
    # Extract close prices and volumes from the state representation
    close_prices = s[0::6]  # Close prices (20 days)
    volumes = s[4::6]       # Volume (20 days)

    # Calculate daily returns from close prices
    returns = np.diff(close_prices) / close_prices[:-1] if close_prices.size > 1 else np.zeros_like(close_prices)

    # Compute additional market features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(returns)
    downside_risk = compute_downside_risk(returns)
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)
    z_score = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Construct the updated state with additional features
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
    # Extract features from the updated state
    realized_volatility = updated_s[120]  # Realized Volatility
    downside_risk = updated_s[121]         # Downside Risk
    relative_momentum = updated_s[126]     # Relative Momentum
    z_score = updated_s[124]               # Z-score for assessing price dynamics
    
    # Assume we can determine current market conditions
    market_regime = 'Defensive'  # Example condition, might be dynamically determined in practice
    threshold_volatility = 0.03  # Risk thresholds for realized volatility
    threshold_downside = 0.03     # Risk thresholds for downside risk
    epsilon = 1e-8  # Small value to prevent division by zero

    if market_regime == 'Defensive':
        # Calculate risk penalties for elevated risks in a defensive market
        risk_penalty = max(0, realized_volatility - threshold_volatility) + max(0, downside_risk - threshold_downside)
        reward = -risk_penalty  # Penalizes higher risk states
        
        # Reward for mean reversion if z-score favors it (means prices are moving back to the average)
        if z_score < 0:
            reward += abs(z_score) * 10  # Incentivize mean-reverting signals

    else:
        # Favorable conditions reward the exploration based on relative momentum
        reward = relative_momentum * 100  # Scale momentum to reward exploration

    return float(reward)  # Return the reward as a float for compatibility with RL framework