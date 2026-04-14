import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    
    # Feature 1: Price Momentum
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Volatility (Standard Deviation of last 20 closing prices)
    volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0
    
    # Feature 3: Average Volume Change
    volumes = s[4::6]  # Extract trading volumes (every 6th element starting from index 4)
    if len(volumes) >= 10:
        avg_volume_last_5 = np.mean(volumes[-5:])
        avg_volume_previous_5 = np.mean(volumes[-10:-5])
        volume_change = (avg_volume_last_5 - avg_volume_previous_5) / avg_volume_previous_5 if avg_volume_previous_5 != 0 else 0
    else:
        volume_change = 0

    features = [price_momentum, volatility, volume_change]
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
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # moderate negative reward for BUY signals
    
    # If we are in a safe risk level
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += np.random.uniform(10, 20)  # positive reward for bullish momentum
            elif trend_direction < -0.3:
                reward += np.random.uniform(10, 20)  # positive reward for bearish momentum
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Implement mean-reversion logic
            # Assuming we can identify some oversold/overbought conditions in features
            # For simplicity, let's say if volatility is low, we reward mean reversion
            if volatility_level < 0.3:
                reward += np.random.uniform(5, 15)  # reward for mean reversion strategies
            else:
                reward -= np.random.uniform(5, 15)  # penalty for breakout-chasing in a sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility
    
    return float(reward)