import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Average True Range (ATR) over the last 14 days to measure volatility
    def compute_atr(prices, period=14):
        if len(prices) < period:
            return 0
        high_low = np.max(prices[-period:]) - np.min(prices[-period:])
        return high_low / np.mean(prices[-period:]) if np.mean(prices[-period:]) != 0 else 0
    
    atr = compute_atr(closing_prices[-14:])

    # Feature 2: Modified RSI (Relative Strength Index) to capture rapid changes
    def compute_modified_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        avg_gain = np.mean(gains[-period:]) if np.mean(gains[-period:]) != 0 else 0
        avg_loss = np.mean(losses[-period:]) if np.mean(losses[-period:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    modified_rsi = compute_modified_rsi(closing_prices[-14:])

    # Feature 3: Maximum Drawdown over the last 30 days to measure risk
    max_drawdown = np.max(np.maximum.accumulate(closing_prices[-30:]) - closing_prices[-30:]) \
                          / np.maximum.accumulate(closing_prices[-30:]) if len(closing_prices[-30:]) > 0 else 0

    # Return the new features as a numpy array
    features = [atr, modified_rsi, max_drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[1] > 70:  # High modified RSI indicating possible overbought
            return -50  # Strong penalty for aggressive buying in high risk
        # Mild positive reward for SELL-aligned features
        elif features[1] < 30:  # Low modified RSI indicating possible oversold
            return 10  # Mild reward for selling in high risk

    elif risk_level > 0.4:
        if features[1] > 70:  # High modified RSI indicating possible overbought
            reward -= 20

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Positive momentum
            reward += 20
        elif trend_direction < -0.3 and features[0] < 0:  # Negative momentum
            reward += 20

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold
            reward += 15  # Reward for mean-reversion buy signal
        elif features[1] > 70:  # Overbought
            reward -= 15  # Penalize for mean-reversion sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds