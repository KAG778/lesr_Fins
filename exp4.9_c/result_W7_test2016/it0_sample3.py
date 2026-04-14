import numpy as np

def revise_state(s):
    # s: 120d raw state
    n_days = 20
    closing_prices = s[0::6][:n_days]  # Extract closing prices
    volumes = s[4::6][:n_days]  # Extract trading volumes

    # Feature 1: Price Momentum (current closing price - closing price 5 days ago)
    momentum = closing_prices[-1] - closing_prices[-6] if n_days > 5 else 0

    # Feature 2: Average Trading Volume over the last 5 days
    avg_volume = np.mean(volumes[-5:]) if n_days >= 5 else np.mean(volumes)

    # Feature 3: Volatility (Standard Deviation of closing prices over the last 5 days)
    volatility = np.std(closing_prices[-5:]) if n_days >= 5 else 0

    features = [momentum, avg_volume, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)  # MILD POSITIVE for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion features
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        # Penalize breakout-chasing features
        reward -= np.random.uniform(5, 10)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds