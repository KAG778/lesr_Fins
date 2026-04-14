import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Compute new features and return them
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) >= 6 else 0.0

    # Feature 2: Volume Change (percentage change from day 5 to day 0)
    volume_change = ((volumes[0] - volumes[5]) / volumes[5]) if volumes[5] > 0 else 0.0

    # Feature 3: Price Range (high - low over the last 5 days)
    highs = s[2:120:6]            # Extracting high prices
    lows = s[3:120:6]             # Extracting low prices
    price_range = np.max(highs) - np.min(lows)

    # Return new features as a numpy array
    features = [momentum, volume_change, price_range]
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

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive momentum
            reward += trend_direction * features[0] * 10.0
        else:  # Negative momentum
            reward += trend_direction * features[0] * -5.0  # Penalize negative momentum in a trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold
            reward += 10.0  # Positive for buying
        elif features[0] > 0:  # Overbought
            reward += 5.0  # Positive for selling
        else:
            reward += 3.0  # Mild reward for holding in a sideways market

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))