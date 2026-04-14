import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract volumes

    features = []

    # Feature 1: Average True Range (ATR) for volatility measure
    if len(closing_prices) > 1:
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr) if len(tr) > 0 else 0
        features.append(atr)
    else:
        features.append(0)

    # Feature 2: Rate of Change (ROC) of closing prices over the last 10 days
    if len(closing_prices) > 10:
        roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if closing_prices[-11] != 0 else 0
        features.append(roc)
    else:
        features.append(0)

    # Feature 3: Distance from 20-day moving average (normalized)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        distance_from_ma = (closing_prices[-1] - moving_avg) / moving_avg if moving_avg != 0 else 0
        features.append(distance_from_ma)
    else:
        features.append(0)

    # Feature 4: Volume Change Rate (percentage change)
    if len(volumes) > 1 and volumes[-2] != 0:
        volume_change_rate = (volumes[-1] - volumes[-2]) / volumes[-2]
        features.append(volume_change_rate)
    else:
        features.append(0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        momentum_score = features[1]  # Assuming features[1] is the ROC
        if trend_direction > trend_threshold and momentum_score > 0:  # Bullish condition
            reward += 20  # Positive reward for correct bullish position
        elif trend_direction < -trend_threshold and momentum_score < 0:  # Bearish condition
            reward += 20  # Positive reward for correct bearish position

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] < 0:  # If price is below recent moving average
            reward += 15  # Reward for buying in oversold condition
        elif features[2] > 0:  # If price is above recent moving average
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds