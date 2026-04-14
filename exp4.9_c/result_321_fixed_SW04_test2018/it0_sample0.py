import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)

    # Feature 1: Price Momentum (current closing minus previous closing)
    # We handle edge cases by ensuring we don't access out of bounds
    price_momentum = s[6] - s[0]  # Day 1 closing price - Day 0 closing price

    # Feature 2: Average Trading Volume (last 20 days)
    # Calculate average volume over the last 20 days
    average_volume = np.mean(s[4::6])  # Every 6th entry starting from index 4

    # Feature 3: Price Range (high - low) of the most recent day
    price_range = s[19 * 6 + 2] - s[19 * 6 + 3]  # Day 19 high - Day 19 low

    # Handle potential division by zero or other edge cases
    price_range = price_range if price_range != 0 else 1e-6

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
        # Strong negative reward for BUY-aligned features
        reward -= 40.0  # STRONG NEGATIVE for buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0  # MODERATE NEGATIVE for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            # Positive reward for upward features in an uptrend
            reward += features[0] * 10.0  # Price momentum
        else:
            # Positive reward for downward features in a downtrend
            reward += features[0] * 10.0  # Price momentum (negative trend)

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming oversold if price momentum is negative
        if features[0] < 0:  # Price momentum indicates oversold
            reward += 5.0  # Mild positive for buying
        # Assuming overbought if price momentum is positive
        elif features[0] > 0:  # Price momentum indicates overbought
            reward += 5.0  # Mild positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))