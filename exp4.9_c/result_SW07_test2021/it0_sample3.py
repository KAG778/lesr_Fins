import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract every 6th element starting from index 0 (closing prices)
    volumes = s[4:120:6]          # Extract every 6th element starting from index 4 (volumes)
    
    # Feature 1: Price Momentum (Change in closing price over the last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Change from day 14 to day 19 (most recent)

    # Feature 2: Volatility (Standard deviation of closing prices over the last 20 days)
    if len(closing_prices) > 1:
        volatility = np.std(closing_prices)  # Standard deviation of the last 20 closing prices
    else:
        volatility = 0.0  # Handle edge case
    
    # Feature 3: Volume Change (Percentage change in volume from day 14 to day 19)
    if volumes[-6] != 0:
        volume_change = (volumes[-1] - volumes[-6]) / volumes[-6]  # Percentage change in volume
    else:
        volume_change = 0.0  # Handle edge case

    # Return the computed features as a numpy array
    return np.array([price_momentum, volatility, volume_change])

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features (replace with actual logic as needed)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)