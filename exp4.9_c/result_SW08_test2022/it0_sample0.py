import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0

    # Feature 2: Volume Change (current volume - previous volume) / previous volume
    volume_change = (volumes[0] - volumes[1]) / volumes[1] if volumes[1] != 0 else 0

    # Feature 3: Average True Range (ATR) over the last 5 days
    true_ranges = np.maximum(s[2:120:6] - s[3:120:6], s[2:120:6] - s[1:120:6], s[1:120:6] - s[3:120:6])
    atr = np.mean(true_ranges[-5:]) if len(true_ranges) > 5 else 0

    # Compile features into a list
    features = [price_momentum, volume_change, atr]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 40  # Strong penalty
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 15  # Moderate penalty

    # If risk is low, apply the other priorities
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 10  # Positive reward for upward features
            else:
                reward += 10  # Positive reward for downward features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3 and risk_level < 0.3:
            reward += 10  # Reward mean-reversion features

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]