import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # New Feature 1: Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0

    # New Feature 2: Relative Strength Index (RSI) calculation
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get the price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    # New Feature 3: Average True Range (ATR) as a measure of volatility
    if len(closing_prices) >= 14:
        high_low = np.array([s[i] - s[i + 1] for i in range(0, len(s) - 1, 6)])
        atr = np.mean(np.abs(high_low[-14:]))  # Simplified ATR
    else:
        atr = 0

    # New Feature 4: Bollinger Band Width (using a rolling mean and std)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        bb_width = (rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)
    else:
        bb_width = 0

    # New Feature 5: Volume Change (percentage change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] > 0 else 0

    features = [price_momentum, rsi, atr, bb_width, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Dynamic thresholds based on historical std
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(40, 60)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)   # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.random.uniform(15, 25)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.random.uniform(15, 25)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123][1] < 30:  # Oversold condition
            reward += np.random.uniform(10, 20)  # Reward for mean reversion
        elif enhanced_s[123][1] > 70:  # Overbought condition
            reward += np.random.uniform(10, 20)  # Also reward for mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]