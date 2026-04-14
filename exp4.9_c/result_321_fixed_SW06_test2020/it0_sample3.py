import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]   # Extract opening prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract trading volumes
    
    features = []

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(average_volume)

    # Feature 3: Price Range (High - Low) over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices)
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = computed features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        # Mild positive for SELL-aligned features
        reward += 5.0 if features[0] < 0 else 0  # Assuming features[0] indicates recent price change
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10.0 * features[0]  # Reward based on price change percentage
        elif trend_direction < -0.3:  # Downtrend
            reward += 10.0 * (-features[0])  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buy
        elif features[0] > 0.01:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))