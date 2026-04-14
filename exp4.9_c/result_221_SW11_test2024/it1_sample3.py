import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Bollinger Bands (20-day)
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        current_price = closing_prices[-1]
        features.append((current_price - lower_band) / (upper_band - lower_band))  # Normalize distance to bands
    else:
        features.append(0.0)  # Default if not enough data

    # Feature 2: Average True Range (ATR) over 14 days
    if n >= 14:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]  # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])
        features.append(atr)
    else:
        features.append(0.0)

    # Feature 3: Recent Price Action (percentage from moving average)
    if n >= 5:
        recent_ma = np.mean(closing_prices[-5:])
        recent_price_action = (closing_prices[-1] - recent_ma) / recent_ma  # Normalize
        features.append(recent_price_action)
    else:
        features.append(0.0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Determine thresholds based on historical std
    risk_threshold_high = 0.7  # This could be derived from historical data
    risk_threshold_medium = 0.4

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        momentum_score = features[0]  # Assuming features[0] is aligned with trends
        if trend_direction > 0.3 and momentum_score > 0:  # Bullish condition
            reward += 20  # Positive reward for correct bullish position
        elif trend_direction < -0.3 and momentum_score < 0:  # Bearish condition
            reward += 20  # Positive reward for correct bearish position

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # If price is below recent moving average
            reward += 10  # Reward for buying in oversold condition
        elif features[2] > 0:  # If price is above recent moving average
            reward += 10  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds