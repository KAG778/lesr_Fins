import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0:120:6]
    
    # Feature 1: Average True Range (ATR)
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1]))))
    features.append(atr)

    # Feature 2: Bollinger Bands (20-day SMA and Std Dev)
    sma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    features.append(upper_band)
    features.append(lower_band)
    
    # Feature 3: Weighted Moving Average (WMA) for last 10 days
    weights = np.arange(1, 11)
    wma = np.sum(closing_prices[-10:] * weights) / np.sum(weights) if len(closing_prices) >= 10 else 0
    features.append(wma)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    avg_risk = np.mean(risk_level)
    std_risk = np.std(risk_level)
    high_risk_threshold = avg_risk + 1.5 * std_risk
    mid_risk_threshold = avg_risk + 0.5 * std_risk

    # Priority 1 - RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -40  # Strong negative for BUY
        reward += 10   # Mild positive for SELL
    elif risk_level > mid_risk_threshold:
        reward += -20  # Moderate negative for BUY
    
    # Priority 2 - TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level <= mid_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Reward for upward momentum
        else:
            reward += 20  # Reward for downward momentum

    # Priority 3 - SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion

    # Priority 4 - HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the bounds of [-100, 100]