import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (current price vs price 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0.0

    # Feature 2: Volume Change (current volume vs average volume of last 5 days)
    current_volume = volumes[0]
    avg_volume = np.mean(volumes[:5]) if len(volumes) > 5 else 1.0  # Avoid division by zero
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    # Feature 3: Price Range (high - low over the last 20 days)
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    price_range = np.max(high_prices) - np.min(low_prices)

    features = [price_momentum, volume_change, price_range]
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
        reward -= 40.0  # Strong negative for BUY signals
        # MILD POSITIVE for SELL-aligned features, could use features[1] (volume change) as a proxy
        reward += 5.0 * features[1]
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10.0 * features[0]  # Price momentum should contribute positively
        else:  # Downtrend
            reward += 10.0 * (-features[0])  # Negative price momentum should contribute positively

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        else:  # Overbought condition
            reward -= 5.0  # Penalty for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))