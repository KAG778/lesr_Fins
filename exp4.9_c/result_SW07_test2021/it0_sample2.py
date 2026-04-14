import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]  # Extract low prices
    
    # Calculate price momentum
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    
    # Calculate average trading volume
    average_volume = np.mean(trading_volumes) if np.sum(trading_volumes) > 0 else 0
    
    # Calculate price range (high - low) over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0
    
    # Collect features into a list
    features = [price_momentum, average_volume, price_range]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        # Return MILD POSITIVE reward for SELL-aligned features
        reward += np.random.uniform(5, 10)
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)