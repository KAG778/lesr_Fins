import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices for the last 20 days
    opening_prices = s[1::6]  # Opening prices for the last 20 days
    high_prices = s[2::6]     # High prices for the last 20 days
    low_prices = s[3::6]      # Low prices for the last 20 days
    volumes = s[4::6]         # Trading volumes for the last 20 days

    # Feature 1: Price Momentum (Percentage change from opening to closing)
    price_momentum = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] if opening_prices[-1] != 0 else 0

    # Feature 2: Average Volume Change (Percentage change in volume across the last 20 days)
    avg_volume = np.mean(volumes)
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Feature 3: Price Range (Difference between high and low prices)
    price_range = high_prices[-1] - low_prices[-1]

    # Return the computed features as a numpy array
    return np.array([price_momentum, volume_change, price_range])

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
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Positive reward for upward momentum
        else:  # Downtrend
            reward += -features[0] * 10.0  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Buy signal
        else:  # Overbought condition
            reward += -5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Return the reward clipped to the range [-100, 100]
    return float(np.clip(reward, -100, 100))