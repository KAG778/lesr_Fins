import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Compute features that help in decision making
    features = []
    
    # Feature 1: Price change percentage over the last 5 days
    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) >= 6:
        price_change = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        price_change = 0  # Default to 0 if not enough data
    features.append(price_change)

    # Feature 2: Moving Average (MA) over the last 5 days
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # Use the last price if not enough data
    features.append(moving_average)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gains / losses if losses > 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Default RSI if not enough data
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming feature 0 indicates bullish signal
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming feature 0 indicates bearish signal
            reward += np.random.uniform(5, 10)
        return reward
    
    if risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= 15  # Arbitrary value for moderate penalty

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish
            reward += 20  # Positive reward for correct bullish signal
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish
            reward += 20  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 15  # Positive for buying in oversold
        elif features[2] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 15  # Positive for selling in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return reward