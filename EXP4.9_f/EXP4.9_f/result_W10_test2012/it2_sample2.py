import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # 1. Price Momentum (latest closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # 2. Relative Strength Index (RSI) calculation over the last 14 days
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if insufficient data

    # 3. Average True Range (ATR) as a measure of volatility
    if len(closing_prices) > 14:
        high_low = np.array([s[i] - s[i + 1] for i in range(0, len(s) - 1, 6)])
        high_close = np.abs(high_low)  # Simplified ATR for demonstration
        atr = np.mean(high_close[-14:]) if len(high_close) >= 14 else 0
    else:
        atr = 0

    # 4. Bollinger Bands: Calculate the width of the bands
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        bb_width = (rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)
    else:
        bb_width = 0

    # 5. Percentage of Price Above 50-Day Moving Average
    if len(closing_prices) >= 50:
        moving_avg_50 = np.mean(closing_prices[-50:])
        price_above_ma50 = ((closing_prices[-1] - moving_avg_50) / moving_avg_50) * 100
    else:
        price_above_ma50 = 0

    features = [price_momentum, rsi, atr, bb_width, price_above_ma50]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)   # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        features = enhanced_s[123:]
        if trend_direction > trend_threshold and features[0] > 0:  # Uptrend alignment
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold and features[0] < 0:  # Downtrend alignment
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[1] < 30:  # Oversold condition
            reward += 15  # Reward for mean reversion
        elif features[1] > 70:  # Overbought condition
            reward += 15  # Reward for mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]