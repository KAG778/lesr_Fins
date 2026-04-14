import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices

    # Feature 1: Bollinger Bands (20-day period)
    sma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    bollinger_width = upper_band - lower_band

    # Feature 2: Average True Range (ATR)
    hl = high_prices[-20:] - low_prices[-20:]  # High-Low range
    hc = np.abs(high_prices[-20:] - closing_prices[-21:-1])  # High-Close range
    lc = np.abs(low_prices[-20:] - closing_prices[-21:-1])  # Low-Close range
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = np.mean(tr) if len(tr) > 0 else 0

    # Feature 3: Exponential Moving Average (EMA, 20-day)
    ema = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0  # Simplified EMA calculation

    return np.array([bollinger_width, atr, ema])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Define relative thresholds based on historical data
    historical_std_dev = np.std(features) if len(features) > 0 else 1
    rsi_threshold_high = 70
    rsi_threshold_low = 30

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 20)  # Moderate penalty for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += (features[0] / historical_std_dev)  # Reward for alignment with momentum
        elif trend_direction < 0:  # Downtrend
            reward += (features[0] / historical_std_dev)  # Reward for downward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < rsi_threshold_low:  # Oversold condition
            reward += 10  # Reward for buying oversold
        elif features[0] > rsi_threshold_high:  # Overbought condition
            reward -= 10  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds