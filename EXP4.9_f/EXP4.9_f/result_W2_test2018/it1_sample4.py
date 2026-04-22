import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Average True Range (ATR) for volatility
    def true_range(prices):
        high = prices[1::6]  # High prices
        low = prices[2::6]   # Low prices
        prev_close = prices[:-6]  # Previous close prices
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        return np.mean(tr[-20:]) if len(tr) > 0 else 0

    atr = true_range(s)
    
    # Feature 2: Relative Strength Index (RSI)
    def rsi(data, period=14):
        delta = np.diff(data)
        gain = (delta[delta > 0]).mean() if len(delta[delta > 0]) > 0 else 0
        loss = (-delta[delta < 0]).mean() if len(delta[delta < 0]) > 0 else 0
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))

    rsi_value = rsi(closing_prices[-20:])

    # Feature 3: 20-day moving average
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    features = [atr, rsi_value, moving_avg]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds for risk
    risk_threshold_high = np.std(features) * 1.5
    risk_threshold_medium = np.std(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[1] > 70:  # Assuming RSI > 70 indicates overbought (BUY-aligned feature)
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY
        else:  # Assuming RSI < 30 indicates oversold (SELL-aligned feature)
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[1] > 70:  # BUY-aligned feature
            reward = -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[1] < 70:  # Uptrend and not overbought
            reward += 10  # Positive reward for upward features
        elif trend_direction < 0 and features[1] > 30:  # Downtrend and not oversold
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        elif features[1] > 70:  # Overbought condition
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]