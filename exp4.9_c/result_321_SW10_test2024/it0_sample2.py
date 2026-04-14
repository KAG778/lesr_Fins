import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    price_changes = []
    volume_changes = []
    price_ranges = []

    for i in range(1, 20):  # From day 1 to day 19
        closing_price_prev = s[(i-1)*6 + 0]
        closing_price_curr = s[i*6 + 0]
        volume_prev = s[(i-1)*6 + 4]
        volume_curr = s[i*6 + 4]
        
        # Price Change
        if closing_price_prev != 0:
            price_change = (closing_price_curr - closing_price_prev) / closing_price_prev
        else:
            price_change = 0  # Handle division by zero

        price_changes.append(price_change)

        # Volume Change
        if volume_prev != 0:
            volume_change = (volume_curr - volume_prev) / volume_prev
        else:
            volume_change = 0  # Handle division by zero

        volume_changes.append(volume_change)

        # Price Range
        high_price = s[i*6 + 2]
        low_price = s[i*6 + 3]
        if closing_price_curr != 0:
            price_range = (high_price - low_price) / closing_price_curr
        else:
            price_range = 0  # Handle division by zero

        price_ranges.append(price_range)

    # Aggregate features: averaging over the last 19 days
    features = [
        np.mean(price_changes),  # Average price change
        np.mean(volume_changes),  # Average volume change
        np.mean(price_ranges)     # Average price range
    ]

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

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Using average price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition (negative price change)
            reward += 5.0  # Mild positive for buying
        elif features[0] > 0.01:  # Overbought condition (positive price change)
            reward -= 5.0  # Mild negative for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))