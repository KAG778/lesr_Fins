import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices and volumes for feature computation
    closing_prices = s[0::6]  # Closing prices at indices 0, 6, 12, ..., 114
    volumes = s[4::6]         # Trading volumes at indices 4, 10, 16, ..., 114

    # Feature 1: Percentage Change in Closing Price
    price_change = closing_prices[-1] / closing_prices[-2] - 1 if closing_prices[-2] != 0 else 0

    # Feature 2: Percentage Change in Trading Volume
    volume_change = volumes[-1] / volumes[-2] - 1 if volumes[-2] != 0 else 0

    # Feature 3: 5-Day Simple Moving Average of Closing Prices
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # Fallback to the last price if there are not enough days

    # Return the features as a 1D numpy array
    return np.array([price_change, volume_change, moving_average])

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
        # Strong negative reward for BUY-aligned features
        reward -= 40.0
        # MILD POSITIVE reward for SELL-aligned features
        reward += 5.0 * features[0]  # Assuming features[0] relates to selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Reward based on price change
        else:  # Downtrend
            reward += -features[0] * 10.0  # Reward based on negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion features
        if features[0] < 0:  # Oversold
            reward += 5.0
        elif features[0] > 0:  # Overbought
            reward += -5.0  # Penalize breakout chasing

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50%
        reward *= 0.5

    return float(np.clip(reward, -100, 100))