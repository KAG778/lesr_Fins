import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    
    # Feature 1: Bollinger Bands (upper and lower)
    window = 20
    if len(closing_prices) >= window:
        sma = np.mean(closing_prices[-window:])
        std = np.std(closing_prices[-window:])
        bollinger_upper = sma + (std * 2)
        bollinger_lower = sma - (std * 2)
    else:
        bollinger_upper = bollinger_lower = np.nan

    # Feature 2: Average True Range (ATR)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    atr = np.mean(np.abs(np.diff(closing_prices[-window:]))) if len(closing_prices) >= window else np.nan

    # Feature 3: Cumulative Return
    cumulative_return = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] * 100 if closing_prices[0] != 0 else 0

    features = [bollinger_upper, bollinger_lower, atr, cumulative_return]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds based on standard deviation
    historical_volatility = np.std(enhanced_s[0:120:6])  # Volatility of closing prices
    threshold_high_risk = 0.7 * historical_volatility
    threshold_low_risk = 0.4 * historical_volatility
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > threshold_high_risk:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
    elif risk_level > threshold_low_risk:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < threshold_low_risk:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for BUY
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for SELL

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 10)  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < threshold_low_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)