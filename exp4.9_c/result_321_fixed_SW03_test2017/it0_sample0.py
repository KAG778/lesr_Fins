import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    # Compute features based on the raw state
    features = []

    # Compute price change: change in closing price from the previous day
    closing_prices = s[0::6]  # Extract closing prices
    price_change = np.diff(closing_prices)  # Daily price changes
    features.append(price_change[-1] if len(price_change) > 0 else 0)  # Last price change

    # Compute volume change: change in volume from the previous day
    volumes = s[4::6]  # Extract trading volumes
    volume_change = np.diff(volumes)  # Daily volume changes
    features.append(volume_change[-1] if len(volume_change) > 0 else 0)  # Last volume change

    # Compute price range: high price - low price for the last day
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your computed features
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
        if len(features) > 0:
            # Use price change feature to inform reward
            reward += trend_direction * features[0] * 10.0  # Price change multiplier

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (assuming oversold condition based on price change)
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Mild positive for mean-reversion BUY
        else:  # Overbought condition
            reward -= 5.0  # Mild negative for mean-reversion SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))