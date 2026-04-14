import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Volumes

    features = []

    # Feature 1: Exponential Moving Average (EMA) - 20 days
    if len(closing_prices) >= 20:
        ema_20 = np.mean(closing_prices[-20:])
    else:
        ema_20 = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(ema_20)

    # Feature 2: Bollinger Bands - Bandwidth (Upper - Lower)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        band_width = upper_band - lower_band
    else:
        band_width = 0
    features.append(band_width)

    # Feature 3: Average True Range (ATR) - 14 days (Volatility Measure)
    if len(closing_prices) >= 14:
        tr = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                        np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                   closing_prices[:-1] - closing_prices[1:]))
        atr = np.mean(tr[-14:])
    else:
        atr = 0
    features.append(atr)

    # Feature 4: Rate of Change (ROC) of closing prices
    if len(closing_prices) > 1:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        if features[3] < 0:  # If the most recent ROC is negative, mild positive for SELL
            reward += np.random.uniform(5, 10)
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level <= risk_threshold_medium:
        if trend_direction > 0:
            reward += features[3] * 20  # Positive reward for momentum in uptrend
        elif trend_direction < 0:
            reward += -features[3] * 20  # Positive reward for momentum in downtrend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -0.02:  # Assuming ROC < -0.02 indicates oversold
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 0.02:  # Assuming ROC > 0.02 indicates overbought
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds