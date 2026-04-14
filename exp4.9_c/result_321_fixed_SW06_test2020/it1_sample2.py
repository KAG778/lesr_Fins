import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Moving Average (20-day)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0

    # Feature 2: Price Change Percentage (Last Day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 3: Average True Range (ATR) for Volatility Estimation
    true_ranges = np.abs(np.diff(closing_prices)) if len(closing_prices) > 1 else np.array([0])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # 14-day ATR

    # Feature 4: Volume Change (Percentage Change from Previous Day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0

    return np.array([moving_average, price_change_pct, atr, volume_change])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        # Mild positive for SELL-aligned features
        reward += 5.0 if features[1] < 0 else 0  # Assuming feature[1] indicates recent negative price change
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[1] > 0:  # Positive price change percentage
            reward += 10.0 * features[1]  # Reward for aligning with trend
        else:  # Negative price change percentage
            reward += -10.0 * features[1]  # Penalize for contrarian position

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < -0.01:  # Oversold condition
            reward += 5.0  # Encourage buying
        elif features[1] > 0.01:  # Overbought condition
            reward += -5.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))