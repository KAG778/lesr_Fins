import numpy as np

def revise_state(s):
    # Extract closing prices from the raw state
    closing_prices = s[0::6]  # every 6th element starting from index 0
    
    # Feature 1: Price Change (%)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    # Feature 2: Average Volume (last 20 days)
    volumes = s[4::6]  # Extract volumes
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    # Feature 3: Bollinger Bands Width
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])  # 20-day Simple Moving Average
        std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
        bollinger_band_width = std_dev * 2  # Width of the Bollinger Bands
    else:
        bollinger_band_width = 0

    # Feature 4: Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) > 6 else 0

    # Feature 5: Average True Range (ATR) for volatility
    if len(closing_prices) > 14:
        atr = np.mean(np.abs(np.diff(closing_prices[-14:])))
    else:
        atr = 0

    # Feature 6: Distance from 20-day Moving Average
    distance_from_ma = closing_prices[-1] - sma if len(closing_prices) >= 20 else 0

    features = [price_change, average_volume, bollinger_band_width, roc, atr, distance_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from `revise_state`
    reward = 0.0

    # Calculate dynamic thresholds based on historical standard deviations of features
    historical_std = np.std(features)  # Standard deviation of all features
    price_change_threshold = historical_std * 1.5
    volume_threshold = historical_std * 1.5
    momentum_threshold = historical_std * 1.5

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        reward += -40 if features[0] > price_change_threshold else 5  # Strong negative for buying in high risk
    elif risk_level > 0.4:
        reward += -20 if features[0] > price_change_threshold else 0  # Moderate negative for buying in elevated risk

    # **Priority 2 — TREND FOLLOWING**
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[3] > momentum_threshold:  # Align with upward momentum
            reward += 15
        elif trend_direction < -0.3 and features[3] < -momentum_threshold:  # Align with downward momentum
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[5] < -price_change_threshold:  # Oversold condition
            reward += 10  # Reward for buying in oversold conditions
        elif features[5] > price_change_threshold:  # Overbought condition
            reward += 10  # Reward for selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds