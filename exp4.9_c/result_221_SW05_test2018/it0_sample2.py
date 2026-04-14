import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Price Change (percentage change from previous day)
    price_change = np.zeros(days - 1)
    for i in range(1, days):
        if closing_prices[i - 1] != 0:  # Avoid division by zero
            price_change[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]

    # Feature 2: Moving Average (last 5 days)
    moving_average = np.zeros(days)
    for i in range(days):
        if i < 4:
            moving_average[i] = np.nan  # Not enough data for the first 4 days
        else:
            moving_average[i] = np.mean(closing_prices[i-4:i+1])

    # Feature 3: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)

    # Combine features, ignoring the first few elements where applicable
    features = []
    features.extend(price_change)  # Add price change feature
    features.extend(moving_average)  # Add moving average feature
    features.append(rsi)  # Add RSI feature

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
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)