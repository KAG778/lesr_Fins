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
    # Extract close prices and volumes from the current state representation
    close_prices = s[0::6]  # close prices every 6 dimensions starting from index 0
    volumes = s[4::6]       # volumes every 6 dimensions starting from index 4

    # Calculate derived features
    # 1. Relative momentum
    relative_momentum = np.array([compute_relative_momentum(close_prices[i:i+20]) for i in range(len(close_prices) - 20 + 1)])
    
    # 2. Realized volatility
    daily_returns = np.diff(close_prices) / close_prices[:-1]  # Calculate daily returns
    realized_volatility = compute_realized_volatility(daily_returns, window=20)

    # 3. Downside risk
    downside_risk = compute_downside_risk(daily_returns, window=20)

    # 4. Multi-horizon momentum
    multi_horizon_momentum = compute_multi_horizon_momentum(close_prices)

    # 5. Z-score of current price vs. N-day mean
    zscore_price = compute_zscore_price(close_prices, window=20)

    # 6. Mean reversion signal
    mean_reversion_signal = compute_mean_reversion_signal(close_prices, window=20)

    # 7. Turnover ratio (latest volume / average volume)
    turnover_ratio = compute_turnover_ratio(volumes, window=20)

    # Concatenate all calculated features with the original state
    updated_s = np.concatenate((s, 
                                 relative_momentum, 
                                 [realized_volatility],
                                 [downside_risk],
                                 multi_horizon_momentum,
                                 [zscore_price],
                                 [mean_reversion_signal],
                                 [turnover_ratio]
                                ))

    return updated_s

def intrinsic_reward(updated_s):
    # Assuming current market regime is 'Defensive'
    # Define constants for reward calculation
    alpha = 1.0
    high_volatility_threshold = 0.05  # example threshold for volatility

    # Extract risk features from updated state representation
    realized_volatility = updated_s[120]  # derived from `revise_state`
    downside_risk = updated_s[121]  # derived from `revise_state`

    # Reward calculation based on current market regime
    if realized_volatility > high_volatility_threshold:
        # Penalizing states with high realized volatility in a defensive regime
        reward = -alpha * max(0, realized_volatility - high_volatility_threshold)
    else:
        # In less risky conditions, reward informative features
        information_signal = updated_s[128]  # example: mean reversion signal
        reward = information_signal  # Reward the strength of mean reversion

    return reward