import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price change over the last 5 days (percentage)
    price_change = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Average trading volume over the last 5 days
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        avg_gain = np.mean(gains[-period:]) if np.mean(gains[-period:]) != 0 else 0
        avg_loss = np.mean(losses[-period:]) if np.mean(losses[-period:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices[-14:])

    # Return features as a numpy array
    features = [price_change, avg_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features and mild positive for SELL-aligned features
        if features[0] > 0:  # Assume feature[0] is aligned with BUY
            return np.random.uniform(-50, -30)
        else:  # Assume feature[1] is aligned with SELL
            return np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assume feature[0] is aligned with BUY
            reward += np.random.uniform(-20, -10)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward features
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[1] > 0:  # Downward features
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] > 70:  # Overbought condition, sell signal
            reward += np.random.uniform(10, 20)
        elif features[2] < 30:  # Oversold condition, buy signal
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified bounds