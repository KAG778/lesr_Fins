import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Compute features
    closing_prices = s[::6]  # Every 6th element starting from index 0
    volumes = s[4::6]        # Every 6th element starting from index 4
    high_prices = s[3::6]    # Every 6th element starting from index 3
    low_prices = s[2::6]      # Every 6th element starting from index 2

    # Feature 1: Price Momentum (last day - day before)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    
    # Feature 2: Average Trading Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Price Range (max high - min low)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0.0
    
    features = [price_momentum, average_volume, price_range]
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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Price momentum positive
            reward += 10.0 * features[0]  # Positive reward for upward momentum
        if features[0] < 0:  # Price momentum negative
            reward += 10.0 * -features[0]  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Buy signal
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))