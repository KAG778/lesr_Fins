import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate features based on the raw state (OHLCV)
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Price Change (percentage change from previous day)
    price_change = np.zeros(20)
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:  # Avoid division by zero
            price_change[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
        else:
            price_change[i] = 0.0

    # Feature 2: Volume Change (percentage change from previous day)
    volume_change = np.zeros(20)
    for i in range(1, 20):
        if volumes[i - 1] != 0:  # Avoid division by zero
            volume_change[i] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]
        else:
            volume_change[i] = 0.0

    # Feature 3: Volatility (standard deviation of closing prices)
    volatility = np.std(closing_prices)

    # Return only the computed features
    return np.array([price_change[-1], volume_change[-1], volatility])  # Return latest values

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = computed features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Price is increasing
            reward += 10.0 * features[0]  # Positive reward for upward trend
        elif features[0] < 0:  # Price is decreasing
            reward += 10.0 * -features[0]  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold situation
            reward += 5.0  # Reward for potential buying opportunity
        elif features[0] > 0:  # Overbought situation
            reward += 5.0  # Reward for potential selling opportunity

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))