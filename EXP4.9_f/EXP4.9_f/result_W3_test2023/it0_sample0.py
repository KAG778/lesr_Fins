import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[::6]  # Extract closing prices (0, 6, 12, ...)
    volumes = s[4::6]        # Extract trading volumes (4, 10, 16, ...)
    
    # Price Momentum (current closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Most recent vs. 5 days ago
    
    # Moving Average of last 5 days
    moving_average = np.mean(closing_prices[-5:])  # Average of last 5 closing prices
    
    # Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] if volumes[-2] != 0 else 1)  # Avoid division by zero
    
    features = [
        price_momentum,
        moving_average,
        volume_change
    ]
    
    return np.array(features)

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
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative reward for BUY signals

    # If risk management doesn't apply, check other priorities
    if risk_level <= 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10  # Positive reward for upward features
            else:
                reward += 10  # Positive reward for downward features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3 and risk_level < 0.3:
            # Assuming we have features that indicate mean-reversion
            reward += 5  # Encourage mean-reversion signals (this is a placeholder)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the valid range
    reward = np.clip(reward, -100, 100)
    
    return reward