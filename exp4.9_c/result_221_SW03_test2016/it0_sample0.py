import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Price Momentum (change from day i-1 to day i)
    price_momentum = closing_prices[-1] - closing_prices[-2]  # Most recent day - previous day

    # Feature 2: Average Volume over the last 5 days
    if len(volumes) >= 5:
        avg_volume = np.mean(volumes[-5:])  # Average of last 5 days
    else:
        avg_volume = np.nan  # Handle edge case if there are not enough days

    # Feature 3: Price Change Percentage
    if s[1] != 0:  # Opening price of the most recent day
        price_change_pct = (closing_prices[-1] - s[1]) / s[1]  # (closing - opening) / opening
    else:
        price_change_pct = np.nan  # Handle division by zero

    features = [price_momentum, avg_volume, price_change_pct]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative reward for BUY-aligned features
        reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        return reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = np.random.uniform(-10, -5)  # Moderate negative reward for BUY
        return reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 30)  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 30)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion features (e.g., oversold→buy)
        # Penalize breakout-chasing features, if any (not implemented here)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds