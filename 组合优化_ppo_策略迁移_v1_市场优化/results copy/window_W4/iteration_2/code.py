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
    # Extract close prices and volumes for the 20-day window
    close_prices = s[0::6]  # Close prices
    volumes = s[4::6]        # Volumes

    # Calculate additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(np.diff(close_prices))  # Daily returns
    downside_risk = compute_downside_risk(np.diff(close_prices))  # Daily returns
    multi_horizon_mo = compute_multi_horizon_momentum(close_prices)  # 5, 10, 20-day momentum
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)

    # Create updated state
    updated_s = np.concatenate([
        s,  # original state
        np.array([relative_momentum, realized_volatility, downside_risk]),  # risk and momentum features
        multi_horizon_mo,                 # multi-horizon momentum (3 features)
        np.array([zscore_price, mean_reversion_signal, turnover_ratio])  # additional features
    ])

    return updated_s

def intrinsic_reward(updated_s):
    # Extracting relevant features from the updated state
    realized_volatility = updated_s[120]  # Realized volatility
    downside_risk = updated_s[121]  # Downside risk
    relative_momentum = updated_s[122]  # Relative momentum
    risk_level = 0.07  # Assuming a market regime risk level from provided context

    # Determine the market regime
    current_market_regime = "balanced"  # Can vary depending on environment monitoring

    # Set the reward based on market conditions
    if current_market_regime == "balanced":
        # Encourage exploration by rewarding higher relative momentum
        reward = relative_momentum
    else:
        # Penalize high risk in unfavorable regimes
        risk_threshold = 0.02  # Define a threshold for the penalty
        reward = -max(0, realized_volatility - risk_threshold) * 0.5  # Scaling the penalty for stability purpose

    # Adding a stability component based on downside risks
    if downside_risk > 0.03:
        reward -= 0.1 * downside_risk  # Penalty based on downside risk

    return reward