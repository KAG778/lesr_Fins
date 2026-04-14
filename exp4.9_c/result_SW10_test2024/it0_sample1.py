import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Close prices are found at indices 0, 6, 12, ..., 114 (20 days)
    close_prices = s[0:120:6]  # Extract closing prices
    volume = s[4:120:6]  # Extract trading volumes
    
    # Feature 1: Price Momentum (normalized)
    price_momentum = (close_prices[0] - close_prices[5]) / (close_prices[5] if close_prices[5] != 0 else 1)
    
    # Feature 2: Volume Change (percentage change)
    volume_change = (volume[0] - volume[1]) / (volume[1] if volume[1] != 0 else 1)
    
    # Feature 3: Volatility (standard deviation of close prices over the last 5 days)
    volatility = np.std(close_prices[:5])  # Standard deviation of the first five days for volatility measure
    
    features = [price_momentum, volume_change, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0  # Initialize reward

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Example strong negative reward
        # MILD POSITIVE reward for SELL-aligned features
        reward += 5  # Example mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for upward features
        else:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Positive reward for mean-reversion features
        reward += -5  # Penalty for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward stays within the limits [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward