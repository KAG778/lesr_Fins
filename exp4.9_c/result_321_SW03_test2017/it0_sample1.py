import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Price change from the previous day
    price_change = np.zeros(19)  # Only 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:
            price_change[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
        else:
            price_change[i - 1] = 0  # Handle division by zero

    # Feature 2: Volume change from the previous day
    volume_change = np.zeros(19)  # Only 19 changes for 20 days
    for i in range(1, 20):
        if volumes[i - 1] != 0:
            volume_change[i - 1] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]
        else:
            volume_change[i - 1] = 0  # Handle division by zero

    # Feature 3: 5-day moving average of closing prices
    moving_average = np.zeros(20)
    for i in range(4, 20):
        moving_average[i] = np.mean(closing_prices[i-4:i+1])  # Average of last 5 days

    # Concatenate features
    features = np.concatenate((price_change, volume_change, moving_average))
    return features

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY-aligned features
        return reward
    if risk_level > 0.4:
        reward += np.random.uniform(-10, -5)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Assuming we have features that indicate upward/downward alignment
        if trend_direction > 0.3:
            reward += 10  # Positive reward for correct bullish signals
        else:
            reward += 10  # Positive reward for correct bearish signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here we would ideally check for mean reversion features
        reward += 5  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)