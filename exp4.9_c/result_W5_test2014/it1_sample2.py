import numpy as np

def revise_state(s):
    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = s[0] - s[6]  # Current close - Close 6 days ago

    # Feature 2: Bollinger Bands (20-day SMA and standard deviation)
    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])  # 20-day Simple Moving Average
        std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
    else:
        upper_band, lower_band = 0, 0

    # Feature 3: Average True Range (ATR) for volatility
    atr = np.mean(np.abs(np.diff(closing_prices[-14:]))) if len(closing_prices) > 14 else 0

    features = [price_momentum, upper_band, lower_band, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical thresholds for dynamic risk assessment
    risk_threshold_high = 0.7  # Example high risk threshold
    risk_threshold_moderate = 0.4  # Example moderate risk threshold
    trend_threshold_high = 0.3  # Example trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40 if features[0] > 0 else 5  # Strong negative for BUY, mild positive for SELL
    elif risk_level > risk_threshold_moderate:
        reward += -20 if features[0] > 0 else 0  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > trend_threshold_high and risk_level < risk_threshold_moderate:
        if trend_direction > trend_threshold_high and features[0] > 0:  # Positive reward for alignment
            reward += 15
        elif trend_direction < -trend_threshold_high and features[0] < 0:  # Positive reward for alignment
            reward += 15

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < 0.3:
        if features[1] < features[2]:  # Assuming lower band is below upper band for mean-reversion
            reward += 10  # Reward for mean-reversion buy
        elif features[1] > features[2]:  # Assuming upper band is above lower band
            reward += 10  # Reward for mean-reversion sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure final reward is within bounds
    return float(np.clip(reward, -100, 100))