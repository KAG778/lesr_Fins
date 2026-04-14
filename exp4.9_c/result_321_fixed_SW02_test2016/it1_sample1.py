import numpy as np

def revise_state(s):
    features = []
    
    # Extract necessary closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Relative Price Change (compared to the last 5 days)
    if len(closing_prices) >= 6:
        recent_price_change = (closing_prices[-1] - np.mean(closing_prices[-6:-1])) / np.mean(closing_prices[-6:-1]) if np.mean(closing_prices[-6:-1]) != 0 else 0
        features.append(recent_price_change)
    else:
        features.append(0)

    # Feature 2: Volume Change (compared to the last 5 days)
    if len(volumes) >= 6:
        recent_volume_change = (volumes[-1] - np.mean(volumes[-6:-1])) / np.mean(volumes[-6:-1]) if np.mean(volumes[-6:-1]) != 0 else 0
        features.append(recent_volume_change)
    else:
        features.append(0)

    # Feature 3: 5-Day Price Momentum (simple moving average difference)
    if len(closing_prices) >= 6:
        momentum = np.mean(closing_prices[-5:]) - closing_prices[-1]  # Compare mean of last 5 to the last price
        features.append(momentum)
    else:
        features.append(0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 10.0 * -features[0]  # Mild positive for SELL-aligned features, if price is decreasing
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Reward for favorable price change
        else:  # Downtrend
            reward += features[0] * 20.0 * -1  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Oversold condition
            reward += 15.0  # Reward for buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 15.0  # Reward for sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))