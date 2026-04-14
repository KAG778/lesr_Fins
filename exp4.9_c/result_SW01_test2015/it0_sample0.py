import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]  # Extracting trading volumes
    
    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change_pct[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    features.append(np.mean(price_change_pct))  # Average of price changes as a feature

    # Feature 2: Simple Moving Average (5-day)
    sma = np.zeros(20)
    for i in range(4, 20):
        sma[i] = np.mean(closing_prices[i-4:i+1])  # 5-day SMA
    
    features.append(sma[19])  # Use the most recent SMA

    # Feature 3: Volume Change Percentage
    volume_change_pct = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_change_pct[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    
    features.append(np.mean(volume_change_pct))  # Average of volume changes as a feature

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early exit
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        else:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features are designed to capture mean-reversion (not detailed here)
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward