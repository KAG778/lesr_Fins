import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (day 0, 1, ..., 19)
    volumes = s[4:120:6]          # Extract volumes (day 0, 1, ..., 19)
    
    # Feature 1: Price Change Percentage
    if closing_prices[-1] != 0:  # Avoid division by zero
        price_change_pct = (closing_prices[-1] - closing_prices[0]) / closing_prices[0]
    else:
        price_change_pct = 0  # Set to zero if the initial price is zero
    
    # Feature 2: Average Volume
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Price Momentum (latest closing price - closing price 10 days ago)
    if len(closing_prices) > 10 and closing_prices[10] != 0:
        momentum = closing_prices[-1] - closing_prices[-11]
    else:
        momentum = 0  # Set to zero if there's not enough data
    
    features = [price_change_pct, avg_volume, momentum]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Negative reward for any buy signals
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7    # Positive reward for sell signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Negative reward for buy signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for upward features
        else:
            reward += 10  # Positive reward for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
        # Penalize breakout-chasing features (could be implemented with specific features)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within [-100, 100]
    reward = max(-100, min(reward, 100))

    return reward