import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices (s[i*6 + 0])
    volumes = s[4::6]        # Extract trading volumes (s[i*6 + 4])

    # Calculate features
    features = []

    # Feature 1: Price Momentum (most recent closing - closing 5 days ago)
    momentum = closing_prices[19] - closing_prices[14]  # Current - 5 days ago
    features.append(momentum)

    # Feature 2: Price Change Ratio (percentage change over the last 5 days)
    if closing_prices[14] != 0:
        price_change_ratio = (closing_prices[19] - closing_prices[14]) / closing_prices[14]
    else:
        price_change_ratio = 0  # Handle division by zero
    features.append(price_change_ratio)

    # Feature 3: Volume Change (percentage change over the last 5 days)
    if volumes[14] != 0:
        volume_change = (volumes[19] - volumes[14]) / volumes[14]
    else:
        volume_change = 0  # Handle division by zero
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime vector
    # enhanced_s[123:] = features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_s[123] > 0:  # Assuming positive features indicate buying
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY
        else:  # Selling or holding
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] > 0:  # Assuming positive features indicate buying
            reward = -10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward = np.random.uniform(10, 30)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward = np.random.uniform(10, 30)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming negative features indicate selling
            reward = np.random.uniform(10, 30)  # Reward mean-reversion features
        else:
            reward = -10  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)