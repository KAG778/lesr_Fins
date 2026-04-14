import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # every 6th element starting from index 0
    volumes = s[4::6]         # every 6th element starting from index 4

    # Calculate features
    price_change = closing_prices[19] - closing_prices[18]  # Most recent day change
    price_momentum = closing_prices[19] - closing_prices[14]  # Momentum over 5 days
    if volumes[18] != 0:  # Prevent division by zero
        volume_change = (volumes[19] - volumes[18]) / volumes[18]  # Percentage change
    else:
        volume_change = 0.0  # Default to 0 if past volume is 0

    # Return new features as an array
    features = [price_change, price_momentum, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize the reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for buy-aligned features
        reward += -40  # Arbitrary strong negative reward
    elif risk_level > 0.4:
        # Moderate negative for buy signals
        reward += -20  # Arbitrary moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for buying in an uptrend
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 5  # Arbitrary positive reward for mean-reversion
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    reward = max(-100, min(100, reward))
    
    return reward