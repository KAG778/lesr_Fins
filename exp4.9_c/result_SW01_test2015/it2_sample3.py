import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Exponential Moving Average (EMA) over 10 days
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)  # Smoothing factor for 10-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-5:]) if len(true_ranges) > 5 else 0.0
    features.append(atr)  # Recent ATR

    # Feature 3: Rate of Change (ROC) to capture momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(roc)  # Recent Rate of Change

    # Feature 4: Relative Strength Index (RSI) to assess overbought/oversold conditions
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = -np.where(delta < 0, delta, 0).mean()
    rs = gain / loss if loss else 0
    rsi = 100 - (100 / (1 + rs))  # Calculate RSI
    features.append(rsi)  # Current RSI value

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features from revised state
    reward = 0.0

    # Calculate dynamic thresholds based on historical features
    historical_std = np.std(features)  # Using features std as a proxy for volatility
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for high risk
        # Positive for selling if momentum is against buying
        reward += 10 if features[2] < 0 else 0  
        return float(np.clip(reward, -100, 100))  # Early exit
    
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        reward += 10 * features[2]  # Align reward with momentum direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15 if features[3] < 30 else -15  # Reward for oversold condition with RSI

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits