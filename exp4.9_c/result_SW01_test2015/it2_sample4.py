import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]  # Trading volumes

    # Feature 1: Exponential Moving Average (EMA - 10 days) to capture longer-term trends
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)  # Smoothing factor for 10-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Relative Strength Index (RSI) for momentum analysis
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = -np.where(delta < 0, delta, 0).mean()
    rs = gain / loss if loss else np.inf
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)  # Current RSI value
    
    # Feature 3: Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0.0
    features.append(atr)  # Recent ATR

    # Feature 4: Bollinger Band Width to assess volatility
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # Standard deviation over last 20 days
    features.append(std_dev)  # Standard deviation as a measure of volatility
    
    # Feature 5: Volume Weighted Average Price (VWAP) for volume context
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0
    features.append(vwap)  # Most recent VWAP

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features from revised state
    reward = 0.0

    # Calculate historical thresholds based on feature statistics
    historical_volatility = np.std(features)
    historical_trend = np.mean(features[0])  # Using EMA for trend
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility
    trend_threshold = 0.3 * historical_trend

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for buying in high risk
        reward += 10 if features[0] < features[4] else 0  # Mild positive for selling if price is below VWAP
        return float(np.clip(reward, -100, 100))  # Early exit
    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for buying

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < medium_risk_threshold:
        reward += 20 * (1 if trend_direction > 0 else -1)  # Reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_volatility:
        reward += 15 if features[1] < 30 else -15  # Reward for oversold condition (RSI < 30)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits