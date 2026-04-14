import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0

    # Feature 3: Volatility (Standard deviation of closing prices)
    volatility = np.std(closing_prices)

    # Return the computed features in a numpy array
    features = [price_change_pct, average_volume, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly penalize BUY-aligned features
        reward += -40  # A strong negative reward for buying in dangerous conditions
    elif risk_level > 0.4:
        # Moderate penalty for BUY signals
        reward += -20  # A moderate negative reward for buying in elevated risk conditions

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Positive reward for upward features
            reward += 20  # Reward for buying in an uptrend
        elif trend_direction < -0.3:
            # Positive reward for downward features
            reward += 20  # Reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 10  # Mild reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50%
        reward *= 0.5

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)