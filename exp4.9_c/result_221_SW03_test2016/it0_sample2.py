import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    close_prices = s[0::6]  # every 6th element starting from index 0
    volumes = s[4::6]       # every 6th element starting from index 4

    # Feature 1: Price Change Rate
    price_change_rate = np.zeros(20)
    price_change_rate[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]  # Percentage change
    price_change_rate[0] = 0  # No change for the first day

    # Feature 2: Average Volume
    avg_volume = np.mean(volumes) if np.sum(volumes) > 0 else 0  # Avoid division by zero

    # Feature 3: Price Momentum (using last 3 days as an example)
    price_momentum = np.zeros(20)
    price_momentum[3:] = (close_prices[3:] - close_prices[:-3]) / close_prices[:-3]  # 3-day momentum
    price_momentum[:3] = 0  # No momentum for the first three days

    # Combine features into a single array
    features = [np.mean(price_change_rate), avg_volume] + price_momentum.tolist()
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Assuming we're checking if the strategy suggests a BUY, apply this logic
        # e.g., if action == 0: reward += -40
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume we have mean-reversion features to reward
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clamp reward to the range [-100, 100]
    return np.clip(reward, -100, 100)