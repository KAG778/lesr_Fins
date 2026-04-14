import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Relative Strength Index (RSI)
    price_changes = np.diff(closing_prices)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gain = np.mean(gains[-14:])  # Last 14 days
    avg_loss = np.mean(losses[-14:])  # Last 14 days
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 2: Average True Range (ATR)
    high_low = s[1::6] - s[2::6]  # High - Low
    high_close = np.abs(s[1::6] - s[0::6][:-1])  # High - Previous Close
    low_close = np.abs(s[2::6] - s[0::6][:-1])  # Low - Previous Close
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.mean(tr[-14:])  # Last 14 days
    features.append(atr)

    # Feature 3: Bollinger Bands (20-day SMA and Std Dev)
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day STD
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    features.append(upper_band)
    features.append(lower_band)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Define relative thresholds based on historical volatility
    historical_std = np.std(features)  # Example for relative threshold, can be refined with historical data
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(5, 15)  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate penalty for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20 * (features[0] / 100)  # Scale by RSI
        elif trend_direction < 0:
            reward += 20 * (1 - features[0] / 100)  # Inverse relation for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10 if features[0] < 30 else 0  # Reward for being oversold (RSI < 30)
        reward -= 10 if features[0] > 70 else 0  # Penalty for being overbought (RSI > 70)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits