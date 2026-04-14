import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Average price over the last 20 days
    avg_price = np.mean(closing_prices)
    
    # Feature 2: Price momentum (current price - price 5 days ago)
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    
    # Feature 3: Volume change (current volume / previous volume - 1)
    volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) > 1 and volumes[-2] != 0 else 0
    
    features = [avg_price, momentum, volume_change]
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
        # Strong negative reward for BUY-aligned features
        reward = -40  # Example strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Example moderate negative reward
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward = 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward = 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features (example)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)