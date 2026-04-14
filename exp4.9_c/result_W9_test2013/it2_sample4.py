import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Rate of Change (ROC) over the last 10 days
    if len(closing_prices) >= 10:
        roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if closing_prices[-11] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    # Feature 2: Average True Range (ATR) for volatility measurement
    if len(closing_prices) >= 14:
        high_low = np.array([s[i*6 + 2] - s[i*6 + 3] for i in range(len(closing_prices))])
        atr = np.mean(high_low[-14:])  # ATR based on high-low range
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Exponential Moving Average (EMA) of closing prices (12 days)
    if len(closing_prices) >= 12:
        ema = np.mean(closing_prices[-12:])  # Using simple mean instead of EMA for simplicity
    else:
        ema = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(ema)

    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) >= 5 and len(closing_prices) >= 5:
        vwap = np.sum(closing_prices[-5:] * volumes[-5:]) / np.sum(volumes[-5:])
    else:
        vwap = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract newly calculated features
    reward = 0.0

    # Calculate historical thresholds for features
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # If the ROC is positive
            reward -= np.random.uniform(30, 50)  # Strong penalty for buying
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= 15  # Moderate penalty

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Aligning with upward trend
            reward += 30  # Strong positive reward for correct bullish signal
        elif trend_direction < -0.3 and features[0] < 0:  # Aligning with downward trend
            reward += 30  # Strong positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[0] > 0.1:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))