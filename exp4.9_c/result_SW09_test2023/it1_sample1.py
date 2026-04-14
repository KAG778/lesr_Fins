import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema = np.mean(closing_prices[-5:])  # Last 5 days as a proxy for EMA for simplicity
    features.append(ema)

    # Feature 2: Price Momentum (current closing price - closing price 5 days ago)
    if len(closing_prices) >= 6:
        price_momentum = closing_prices[-1] - closing_prices[-6]  # Current vs. 5 days ago
    else:
        price_momentum = 0
    features.append(price_momentum)

    # Feature 3: Volatility Measurement (standard deviation of the last 5 closing prices)
    if len(closing_prices) >= 5:
        volatility = np.std(closing_prices[-5:])
    else:
        volatility = 0
    features.append(volatility)

    # Feature 4: Volume Momentum (current volume - average volume over the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_momentum = volumes[-1] - avg_volume
    features.append(volume_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk levels
    risk_threshold_high = 0.7  # Replace with historical std analysis if available
    risk_threshold_medium = 0.4  # Replace with historical std analysis if available

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for alignment with upward trend
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for alignment with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward