import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4:120:6]          # Extract trading volumes (every 6th element starting from index 4)
    
    features = []
    
    # Compute Price Momentum (current closing - closing 5 days ago)
    momentum_period = 5
    if len(closing_prices) > momentum_period:
        momentum = closing_prices[0] - closing_prices[momentum_period]
    else:
        momentum = 0  # Handle edge case
    features.append(momentum)

    # Compute Volume Change (current volume - previous volume) / previous volume
    if len(volumes) > 1 and volumes[1] != 0:
        volume_change = (volumes[0] - volumes[1]) / volumes[1]
    else:
        volume_change = 0  # Handle edge case
    features.append(volume_change)

    # Compute Price Range (high - low of the last day)
    high_price = s[19*6 + 2]  # High price of the most recent day
    low_price = s[19*6 + 3]   # Low price of the most recent day
    price_range = high_price - low_price
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0  # Initialize reward

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY features
        reward += np.random.uniform(5, 10)     # MILD POSITIVE reward for SELL features
    elif risk_level > 0.4:
        reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY 
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Return the reward within the specified range