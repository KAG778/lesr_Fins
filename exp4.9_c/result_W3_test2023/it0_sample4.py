import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price Momentum (current - past)
    price_momentum = closing_prices[-1] - closing_prices[-10] if len(closing_prices) > 10 else 0.0

    # Feature 2: Average Volume Change
    if np.mean(volumes[:-1]) != 0:  # Check for division by zero
        avg_volume_change = (volumes[-1] - np.mean(volumes[:-1])) / np.mean(volumes[:-1])
    else:
        avg_volume_change = 0.0

    # Feature 3: Price Volatility (Standard Deviation)
    price_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0

    features = [price_momentum, avg_volume_change, price_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = computed features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += features[0]  # Positive for upward features
        else:
            reward += -features[0] # Positive for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold situation
            reward += 10  # Reward for buy
        else:  # Overbought situation
            reward -= 10  # Penalize for sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip rewards to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward