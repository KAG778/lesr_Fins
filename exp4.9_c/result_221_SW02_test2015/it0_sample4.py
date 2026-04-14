import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract relevant data from the state
    closing_prices = s[::6]  # Closing prices (every 6th element starting from index 0)
    opening_prices = s[1::6]  # Opening prices (every 6th element starting from index 1)
    high_prices = s[2::6]  # High prices (every 6th element starting from index 2)
    low_prices = s[3::6]  # Low prices (every 6th element starting from index 3)
    volumes = s[4::6]  # Trading volumes (every 6th element starting from index 4)
    
    # Compute features
    features = []
    
    # Feature 1: Price Momentum (percentage change)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change over last 5 days (relative change)
    if len(volumes) >= 6:  # Ensure there are enough days to compute the average
        avg_volume_last_5 = np.mean(volumes[-5:])
        avg_volume_previous_5 = np.mean(volumes[-10:-5]) if len(volumes) > 10 else avg_volume_last_5
        volume_change = (avg_volume_last_5 - avg_volume_previous_5) / avg_volume_previous_5 if avg_volume_previous_5 != 0 else 0
        features.append(volume_change)
    
    # Feature 3: Bollinger Bands (20-day rolling standard deviation)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        current_price = closing_prices[-1]
        if current_price > upper_band:
            features.append(1)  # Overbought signal
        elif current_price < lower_band:
            features.append(-1)  # Oversold signal
        else:
            features.append(0)  # Neutral
    else:
        features.append(0)  # Neutral if not enough data

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # This will contain the new features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Assuming a BUY-aligned feature is present, and penalize it
        if features[0] > 0:  # Assuming feature[0] represents a momentum signal suggesting BUY
            reward += -40  # Penalize for risky BUY
        # MILD POSITIVE reward for SELL-aligned features
        if features[0] < 0:  # Assuming feature[0] represents a momentum signal suggesting SELL
            reward += 10  # Reward for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming feature[0] suggests a BUY
            reward += -20  # Penalize for risky BUY
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            if features[0] > 0:  # Assuming feature[0] represents upward momentum
                reward += 20  # Positive reward for following the trend
        else:
            if features[0] < 0:  # Assuming feature[0] represents downward momentum
                reward += 20  # Positive reward for correct bearish bet
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] == -1:  # Assuming -1 indicates oversold
            reward += 15  # Reward for buying in an oversold condition
        elif features[0] == 1:  # Assuming 1 indicates overbought
            reward += -15  # Penalize for buying in an overbought condition
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds