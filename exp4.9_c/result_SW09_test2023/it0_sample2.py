import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: 5-day Moving Average
    moving_average = np.mean(closing_prices[-5:])  # Last 5 days
    # Handle edge case for division by zero
    moving_average = moving_average if moving_average != 0 else np.nan

    # Feature 2: Price Momentum (current closing price - closing price 3 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-4]  # Day 19 - Day 16

    # Feature 3: Volume Change (current volume - previous day's volume) / previous day's volume
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] if volumes[-2] != 0 else np.nan)

    # Return features as a numpy array
    features = [moving_average, price_momentum, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for BUY-aligned features
        reward += -40  # Example strong negative for a BUY signal
        # MILD POSITIVE for SELL-aligned features
        reward += 5  # Example mild positive for a SELL signal
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate negative for a BUY signal

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward += -5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward