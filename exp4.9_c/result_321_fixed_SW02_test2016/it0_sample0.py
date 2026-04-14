import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    features = []

    # Feature 1: Price Change (percentage change)
    price_change = (closing_prices[0] - closing_prices[1]) / closing_prices[1] if closing_prices[1] != 0 else 0
    features.append(price_change)

    # Feature 2: Volume Change (percentage change)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] if volumes[1] != 0 else 0
    features.append(volume_change)

    # Feature 3: Price Momentum (average of the last 5 closing prices)
    if len(closing_prices) >= 5:
        momentum = np.mean(closing_prices[:5]) - closing_prices[5]  # Mean of last 5 days compared to the 6th day
    else:
        momentum = 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for buying
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            if trend_direction > 0:  # Uptrend
                reward += features[0] * 10.0  # Favorable price change
            else:  # Downtrend
                reward += features[1] * 10.0  # Favorable volume change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Mild positive for buy signal in mean-reversion
        elif features[0] > 0.01:  # Overbought condition
            reward += 5.0  # Mild positive for sell signal in mean-reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))