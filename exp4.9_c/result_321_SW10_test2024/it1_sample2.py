import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]
    highs = s[2:120:6]
    lows = s[3:120:6]

    # Feature 1: Normalized Price Momentum
    price_momentum = (closing_prices[-1] - closing_prices[-30]) / (np.std(closing_prices[-30:]) + 1e-10)

    # Feature 2: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / (np.sum(volumes) + 1e-10)

    # Feature 3: Bollinger Band Width
    rolling_mean = np.mean(closing_prices[-20:])
    rolling_std = np.std(closing_prices[-20:])
    bollinger_band_width = (rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)

    return np.array([price_momentum, vwap, bollinger_band_width])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0 * (features[0] if features[0] < 0 else 1)  # Strong negative for BUY signals
        reward += 5.0  # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= 10.0 * (features[0] if features[0] < 0 else 1)  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Align with momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 10.0  # Reward for considering a BUY
        elif features[0] > 0.01:  # Overbought condition
            reward += 10.0  # Reward for considering a SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))