import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]  # Trading volumes

    # Feature 1: Exponential Moving Average (EMA - 5 days)
    alpha = 2 / (5 + 1)
    ema = closing_prices[0]  # Starting EMA with first closing price
    for price in closing_prices[1:]:
        ema = (price - ema) * alpha + ema
    features.append(ema)
    
    # Feature 2: Volatility (Standard Deviation of Closing Prices)
    volatility = np.std(closing_prices)
    features.append(volatility)

    # Feature 3: Volume Change (percentage change from the previous day)
    volume_changes = np.diff(volumes) / volumes[:-1]
    volume_changes = np.concatenate(([0], volume_changes))  # Pad with 0 for same length
    features.append(volume_changes[-1])  # Use the most recent change
    
    # Feature 4: Bollinger Band Width
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # Standard deviation over last 20 days
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    features.append(upper_band - lower_band)  # Bollinger Band Width

    # Feature 5: Average True Range (ATR) for volatility measurement
    atr = np.mean(np.abs(np.diff(closing_prices[-14:])))
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    historical_volatility = np.std(enhanced_s[123:][:5])  # Sample historical volatility
    historical_trend = np.mean(enhanced_s[123:][:5])  # Sample historical trend
    threshold_risk_high = 0.7 * historical_volatility
    threshold_trend_high = 0.3 * historical_trend
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)  # Mild positive for SELL
        return float(np.clip(reward, -100, 100))  # Early exit
    elif risk_level > 0.4 * historical_volatility:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > threshold_trend_high and risk_level < 0.4 * historical_volatility:
        reward += 20  # Positive reward for aligned trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.2 * historical_trend and risk_level < 0.3 * historical_volatility:
        reward += 10  # Positive reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_volatility:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits