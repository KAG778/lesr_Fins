import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element from index 0)
    volumes = s[4::6]  # Extract volumes (every 6th element from index 4)

    # Feature 1: Adaptive Momentum (5-day and 20-day)
    if len(closing_prices) >= 20:
        short_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]  # Last 5 days
        long_momentum = (closing_prices[-1] - closing_prices[-21]) / closing_prices[-21]  # Last 20 days
    else:
        short_momentum = long_momentum = 0

    # Feature 2: Historical Volatility (20-day rolling)
    if len(closing_prices) >= 21:
        returns = np.diff(closing_prices[-20:]) / closing_prices[-21:-1]  # Last 20 returns
        historical_volatility = np.std(returns)
    else:
        historical_volatility = 0

    # Feature 3: Z-score of Returns
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_score = (returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0

    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0
    else:
        vwap = 0

    features = [short_momentum, long_momentum, historical_volatility, z_score, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals
    else:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:  # Uptrend
                reward += np.random.uniform(10, 20)  # Positive for upward features
            else:  # Downtrend
                reward += np.random.uniform(10, 20)  # Positive for downward features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            reward += np.random.uniform(5, 15)  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward stays within bounds of [-100, 100]
    return max(-100, min(100, reward))