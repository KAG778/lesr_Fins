import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Extract closing prices and volume
    closing_prices = s[0:120:6]  # Closing prices for the past 20 days
    volumes = s[4:120:6]          # Trading volumes for the past 20 days

    # Feature 1: Price Change Percentage (current vs previous day)
    price_changes = np.diff(closing_prices) / closing_prices[:-1] * 100  # in percentage
    price_change_percentage = np.append(0, price_changes)  # prepend 0 for the first day
    features.append(price_change_percentage[-1])  # last day's price change

    # Feature 2: Volume Change Percentage (current vs previous day)
    volume_changes = np.diff(volumes) / volumes[:-1] * 100  # in percentage
    volume_change_percentage = np.append(0, volume_changes)  # prepend 0 for the first day
    features.append(volume_change_percentage[-1])  # last day's volume change

    # Feature 3: 5-day Moving Average of Closing Prices
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    moving_average = np.append([0, 0, 0, 0], moving_average)  # prepend zeros for alignment
    features.append(moving_average[-1])  # last moving average

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Define reward variable
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        # Assuming action is BUY
        return reward
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (hypothetical, as features are not specified)
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward