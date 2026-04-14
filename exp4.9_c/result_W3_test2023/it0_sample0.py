import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 (closing prices)
    opening_prices = s[1:120:6]   # Every 6th element starting from index 1 (opening prices)
    high_prices = s[2:120:6]      # Every 6th element starting from index 2 (high prices)
    low_prices = s[3:120:6]       # Every 6th element starting from index 3 (low prices)
    volumes = s[4:120:6]          # Every 6th element starting from index 4 (volumes)

    # Feature 1: Price Momentum (current closing price vs closing price 5 days ago)
    # Handle edge case for less than 5 days of data
    if len(closing_prices) < 6:
        price_momentum = 0  # No momentum can be calculated
    else:
        price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0

    # Feature 3: Price Range (high - low) normalized
    if len(high_prices) < 5 or len(low_prices) < 5:
        price_range = 0
    else:
        price_range = (np.max(high_prices[-5:]) - np.min(low_prices[-5:])) / np.mean(closing_prices[-5:]) if np.mean(closing_prices[-5:]) != 0 else 0

    return np.array([price_momentum, avg_volume, price_range])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Check features for BUY or SELL alignment
        if enhanced_s[123] > 0:  # Assuming feature[0] is aligned with BUY
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # Assuming feature[1] is aligned with SELL
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        reward = np.random.uniform(-10, -5)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Buy aligned with uptrend
            reward += 10  # Positive reward for trend-following BUY
        elif trend_direction < -0.3 and enhanced_s[124] > 0:  # Sell aligned with downtrend
            reward += 10  # Positive reward for trend-following SELL

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold → buy
            reward += 10  # Reward for mean-reversion BUY
        elif enhanced_s[124] > 0:  # Overbought → sell
            reward += 10  # Reward for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]