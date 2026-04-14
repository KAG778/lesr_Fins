import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Compute price momentum
    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) > 1:
        price_momentum = closing_prices[-1] - closing_prices[-2]
    else:
        price_momentum = 0  # Handle edge case
    
    features.append(price_momentum)

    # Compute volume change
    volumes = s[4::6]  # Extract trading volumes
    if len(volumes) > 1 and volumes[-2] > 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0  # Handle edge case

    features.append(volume_change)

    # Compute price range
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    if len(high_prices) > 0 and len(low_prices) > 0:
        price_range = high_prices[-1] - low_prices[-1]
    else:
        price_range = 0  # Handle edge case

    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Example value
        # Mild positive reward for SELL-aligned features
        reward += +7   # Example value
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example value

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward momentum
            reward += 15   # Example value
        elif trend_direction < -0.3 and features[0] < 0:  # Downward momentum
            reward += 15   # Example value

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 10   # Example value
        elif features[0] > 0:  # Overbought condition
            reward += -10   # Example value

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)