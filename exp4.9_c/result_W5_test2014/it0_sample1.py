import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # every 6th element starting from index 0
    volumes = s[4::6]         # every 6th element starting from index 4

    # 1. Price Change: Current close - Previous close
    price_change = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    
    # 2. Average Volume: Average of the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # 3. Price Momentum (5-day): Current close - Close from 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    
    # Return the new features as a numpy array
    features = [price_change, average_volume, price_momentum]
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
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Strong negative reward for buying in dangerous conditions
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate negative reward for buying in elevated risk
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for buying in a strong uptrend
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for selling in a strong downtrend
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume features are provided for mean-reversion
        # Penalize breakout-chasing features (not implemented here but can be done)
        reward += 5  # This is a generic reward for mean-reversion
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Return the final reward, ensure it's within the allowed range
    return float(np.clip(reward, -100, 100))