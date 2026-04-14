import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Price Momentum over the last 5 days
    if n > 5:
        price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    else:
        price_momentum = 0
    features.append(price_momentum)

    # Feature 2: Volume Change Percentage over the last 5 days
    if n > 5:
        volume_change = (volumes[-1] - volumes[-6]) / volumes[-6] if volumes[-6] != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    # Feature 3: Average True Range (ATR) over the last 14 days (volatility measure)
    if n >= 15:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])
    else:
        atr = 0
    features.append(atr)

    # Feature 4: Distance from moving average (20-day MA) as a percentage
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        distance_from_ma = (closing_prices[-1] - moving_average) / moving_average if moving_average != 0 else 0
    else:
        distance_from_ma = 0
    features.append(distance_from_ma)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical dynamic thresholds based on feature std
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If price momentum is negative, mild positive for SELL
            reward += np.random.uniform(5, 10)
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold_medium:
        if trend_direction > trend_threshold and features[0] > 0:  # Bullish condition
            reward += features[0] * 20  # Positive reward for momentum
        elif trend_direction < -trend_threshold and features[0] < 0:  # Bearish condition
            reward += -features[0] * 20  # Positive reward for bearish momentum

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[3] < 0:  # If price is below moving average, reward buying in oversold condition
            reward += 10
        elif features[3] > 0:  # If price is above moving average, reward selling in overbought condition
            reward += 10

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds