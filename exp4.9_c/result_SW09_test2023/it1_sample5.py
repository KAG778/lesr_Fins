import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: 10-day Moving Average
    ma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    features.append(ma_10)

    # Feature 2: Price Momentum (current closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Current vs 5 days ago
    features.append(price_momentum)

    # Feature 3: Volume Momentum (current volume - average volume of last 5 days)
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_momentum = volumes[-1] - avg_volume_5  # Current vs average 5 days
    features.append(volume_momentum)

    # Feature 4: Market Volatility (standard deviation of closing prices over last 10 days)
    volatility = np.std(closing_prices[-10:]) if len(closing_prices) >= 10 else 0
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds
    risk_threshold_high = 0.7  # Example threshold, should be based on historical data
    risk_threshold_medium = 0.4  # Example threshold, should be based on historical data
    trend_threshold_high = 0.3   # Example threshold, should be based on historical data

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features (risk-off)
    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold_high and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold_high:  # Uptrend
            reward += 15  # Reward for upward features
        elif trend_direction < -trend_threshold_high:  # Downtrend
            reward += 15  # Reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold_high and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward