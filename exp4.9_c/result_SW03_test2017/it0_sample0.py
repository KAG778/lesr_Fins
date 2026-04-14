import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Edge case: Ensure we don't run into index errors.
    if len(s) < 120:
        return np.array(features)
    
    # Price Momentum: Current close - Close 5 days ago
    close_today = s[114]  # s[19 * 6 + 0], closing price of the most recent day
    close_5_days_ago = s[84]  # s[14 * 6 + 0], closing price 5 days ago
    price_momentum = close_today - close_5_days_ago
    features.append(price_momentum)

    # Volume Change: (Current volume - Volume 5 days ago) / Volume 5 days ago
    volume_today = s[114 + 4]  # s[19 * 6 + 4], volume of the most recent day
    volume_5_days_ago = s[84 + 4]  # s[14 * 6 + 4], volume 5 days ago
    if volume_5_days_ago != 0:
        volume_change = (volume_today - volume_5_days_ago) / volume_5_days_ago
    else:
        volume_change = 0  # Handle division by zero
    features.append(volume_change)

    # Price Range: High - Low of the most recent day
    high_today = s[114 + 2]  # s[19 * 6 + 2], high price of the most recent day
    low_today = s[114 + 3]  # s[19 * 6 + 3], low price of the most recent day
    price_range = high_today - low_today
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        # Assuming features from model would indicate mean-reversion
        reward += 5  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)