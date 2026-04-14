import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price Change Percentage over the last 10 days
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average Daily Volume Change Percentage over the last 10 days
    if len(volumes) >= 10:
        avg_volume_change = (volumes[-1] - np.mean(volumes[-10:])) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) != 0 else 0
    else:
        avg_volume_change = 0
    features.append(avg_volume_change)

    # Feature 3: Bollinger Bands indicator (distance from the upper band)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)  # 2 standard deviations
        distance_from_upper_band = (closing_prices[-1] - upper_band) / upper_band if upper_band != 0 else 0  # Measure of overbought
    else:
        distance_from_upper_band = 0
    features.append(distance_from_upper_band)

    # Feature 4: Price Momentum (current - previous day)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) >= 2 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Relative thresholds based on historical data
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price change
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 10)    # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Aligning with upward momentum
            reward += 30  # Strong positive reward for bullish signals
        elif trend_direction < -0.3 and features[0] < 0:  # Aligning with downward momentum
            reward += 30  # Strong positive reward for bearish signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -0.1:  # Indicates oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[2] > 0.1:  # Indicates overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(reward, 100))

    return reward