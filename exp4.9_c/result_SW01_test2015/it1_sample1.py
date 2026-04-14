import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and trading volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Exponential Moving Average (EMA) to capture trends
    ema = np.zeros_like(closing_prices)
    alpha = 0.1  # Smoothing factor
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = alpha * closing_prices[i] + (1 - alpha) * ema[i - 1]
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Average True Range (ATR) to measure volatility
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0.0
    features.append(atr)  # Recent ATR

    # Feature 3: Rate of Change (ROC) to capture momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(roc)  # Recent Rate of Change

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Your computed features from revise_state
    reward = 0.0
    
    # Calculate historical thresholds for risk and trend
    mean_volatility = 0.2  # Example mean, should be computed from historical data
    std_volatility = 0.1   # Example std, should be computed from historical data
    threshold_risk_low = mean_volatility - std_volatility
    threshold_risk_high = mean_volatility + std_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        reward -= np.random.uniform(30, 50)  # Strong penalty for high risk
        reward += 10 if features[2] < 0 else 0  # Mild positive if momentum is against buying
    elif risk_level > threshold_risk_low:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for moderate risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < threshold_risk_low:
        reward += 10 * features[2]  # Align with momentum direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10 if features[2] < 0 else 0  # Reward for mean-reversion when oversold

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < threshold_risk_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is in range