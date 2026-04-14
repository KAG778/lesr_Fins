import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]          # Trading volumes for 20 days
    
    # Feature 1: Price Change (%)
    price_changes = np.zeros(20)
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:
            price_changes[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    
    # Feature 2: Volume Change (%)
    volume_changes = np.zeros(20)
    for i in range(1, 20):
        if volumes[i - 1] != 0:
            volume_changes[i] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]
    
    # Feature 3: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        gains = np.where(prices[1:] > prices[:-1], prices[1:] - prices[:-1], 0)
        losses = np.where(prices[1:] < prices[:-1], prices[:-1] - prices[1:], 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)
    
    # Combine features into a single array
    features = np.concatenate([price_changes, volume_changes, np.array([rsi_value])])
    
    return np.array(features)

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
        if enhanced_state[123] == 0:  # Assuming BUY-aligned features
            return np.random.uniform(-50, -30)
        elif enhanced_state[123] == 1:  # Assuming SELL-aligned features
            return np.random.uniform(5, 10)
    elif risk_level > 0.4:
        if enhanced_state[123] == 0:  # Assuming BUY-aligned features
            reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] == 0:  # Assuming oversold BUY
            reward += 15
        elif enhanced_state[123] == 1:  # Assuming overbought SELL
            reward += 15

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range