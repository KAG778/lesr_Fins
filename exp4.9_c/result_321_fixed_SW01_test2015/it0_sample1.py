import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes

    # Calculate features
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    price_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] > 0 else 0.0

    features = [price_momentum, price_volatility, volume_change]
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
        # Mild positive for SELL-aligned features
        reward += 5.0 * abs(features[0])  # Assuming features[0] is related to selling momentum
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += features[0] * 10.0  # Positive reward for price momentum in uptrend
        elif trend_direction < -0.3:  # Downtrend
            reward += -features[0] * 10.0  # Positive reward for negative momentum in downtrend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold
            reward += 10.0  # Buy signal
        elif features[0] > 0:  # Overbought
            reward -= 10.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))