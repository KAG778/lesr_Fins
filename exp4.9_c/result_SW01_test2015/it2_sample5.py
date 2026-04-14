import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]  # Trading volumes

    # Feature 1: Exponential Moving Average (EMA - 10 days)
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Relative Strength Index (RSI) for momentum
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = -np.where(delta < 0, delta, 0).mean()  # Average loss
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)  # Current RSI value

    # Feature 3: Bollinger Band Width (to capture volatility)
    moving_average = np.mean(closing_prices[-20:])  # 20-day moving average
    std_dev = np.std(closing_prices[-20:])           # 20-day standard deviation
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)
    features.append(upper_band - lower_band)  # Width of Bollinger Bands

    # Feature 4: Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0.0
    features.append(atr)  # Recent ATR

    # Feature 5: Volume Change (percentage change from the previous day)
    volume_changes = np.diff(volumes) / volumes[:-1]
    volume_changes = np.concatenate(([0], volume_changes))  # Pad with zero for same length
    features.append(volume_changes[-1])  # Most recent volume change

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Your computed features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical features
    historical_volatility = np.std(features)  # Using features as a proxy for historical volatility
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility
    trend_threshold = 0.3 * np.std(features[0:2])  # Using EMA and RSI to define trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for high risk
        reward += 10 if features[0] < 0 else 0  # Mild positive if EMA is falling (momentum against buying)
        return float(np.clip(reward, -100, 100))  # Early exit

    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for medium risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < medium_risk_threshold:
        reward += 20 if trend_direction > 0 else -20  # Positive reward if aligning with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_volatility:
        reward += 15 if features[1] < 30 else -15  # Reward for mean-reversion if RSI indicates oversold

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits