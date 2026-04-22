import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Change (percentage change)
    price_change = np.zeros(len(closing_prices))
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:  # Prevent division by zero
            price_change[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Volume Change (percentage change)
    volume_change = np.zeros(len(volumes))
    for i in range(1, len(volumes)):
        if volumes[i-1] != 0:  # Prevent division by zero
            volume_change[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    
    # Feature 3: Relative Strength Index (RSI) Calculation
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices) if len(closing_prices) >= 15 else np.nan  # Calculate RSI only if enough data

    # Return features as a numpy array
    return np.array([price_change[-1], volume_change[-1], rsi])

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
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        return reward
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_state[123:]
        if trend_direction > 0.3 and features[0] > 0:  # Price change positive
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3 and features[0] < 0:  # Price change negative
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_state[123:]
        if features[2] < 30:  # Oversold
            reward += 15  # Reward for oversold
        elif features[2] > 70:  # Overbought
            reward += 15  # Reward for overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    return reward