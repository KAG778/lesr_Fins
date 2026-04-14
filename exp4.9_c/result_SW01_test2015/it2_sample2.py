import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Exponential Moving Average (EMA - 10 days) for better trend capture
    ema = np.zeros(len(closing_prices))
    alpha = 2 / (10 + 1)  # Smoothing factor for 10-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Relative Strength Index (RSI - 14 days) for momentum measurement
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-14:])  # Average gain over the last 14 days
    avg_loss = np.mean(loss[-14:])  # Average loss over the last 14 days
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)  # Current RSI value

    # Feature 3: Average True Range (ATR - 14 days) for volatility measurement
    high_prices = s[1::6]  # High prices
    low_prices = s[2::6]   # Low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:])  # ATR over the last 14 days
    features.append(atr)  # Recent ATR

    # Feature 4: Bollinger Band Width to capture volatility and potential breakout
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    bollinger_band_width = (2 * std_dev) / sma if sma != 0 else 0  # Normalize by SMA
    features.append(bollinger_band_width)  # Bollinger Band Width
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features from revised state
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(features)  # Using features as a proxy for historical volatility
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility
    trend_threshold = 0.3 * historical_volatility  # Normalizing trend based on historical data

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for high risk
        reward += 10 if features[0] < 0 else 0  # Mild positive for selling
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for medium risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15 if features[1] < 30 else -15 if features[1] > 70 else 0  # Reward for oversold/overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude in high volatility

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits