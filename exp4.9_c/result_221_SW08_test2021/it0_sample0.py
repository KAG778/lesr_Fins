import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Every 6th element starting from index 0
    volumes = s[4::6]        # Every 6th element starting from index 4
    
    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)  # We can only calculate for days 1 to 19 (19 changes)
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change_pct[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]

    # Feature 2: Moving Average of the last 5 closing prices
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')  # Only last 16 days can be computed
    moving_average = np.concatenate([np.zeros(4), moving_average])  # Prepend zeros for the first 4 days

    # Feature 3: Volume Change Percentage
    volume_change_pct = np.zeros(19)
    for i in range(1, 20):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_change_pct[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    
    # Combine features into a single array
    features = np.hstack((price_change_pct, moving_average, volume_change_pct))
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # Extract regime information
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward