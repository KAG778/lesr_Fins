import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (last 5 days)
    try:
        price_momentum = (s[114] - s[109]) / s[109]  # (Close day 19 - Close day 14) / Close day 14
    except ZeroDivisionError:
        price_momentum = 0.0
    features.append(price_momentum)

    # Feature 2: Average Volume over the last 10 days
    avg_volume = np.mean(s[4::6][:10])  # Average of the last 10 days' trading volume
    features.append(avg_volume)

    # Feature 3: Bollinger Bands (Upper and Lower Bands)
    closing_prices = s[0::6][:20]  # Last 20 closing prices
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices)
        rolling_std = np.std(closing_prices)
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        features.append(upper_band)
        features.append(lower_band)
    else:
        features.extend([0, 0])  # Default values if not enough data

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical volatility
    volatility_threshold = np.std(enhanced_s[123:])  # Use the standard deviation of features as a threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 50  # Strong penalty for buying in high-risk conditions
        reward += 10   # Mild positive reward for selling (risk-off)
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= 0.4:
        if trend_direction > 0:
            reward += 25  # Strong reward for aligning with upward trend
        else:
            reward += 25  # Strong reward for aligning with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.4:
        reward += 15  # Reward mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    # Ensure the reward is within [-100, 100]
    return np.clip(reward, -100, 100)