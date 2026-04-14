import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices for the last 20 days
    volumes = s[4:120:6]          # Extract volumes for the last 20 days
    
    # Feature 1: Price Change (%)
    price_change = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] if closing_prices[0] != 0 else 0
    
    # Feature 2: Average Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Price Momentum (5-day momentum)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Return the computed features as a numpy array
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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume we have some logic to determine the mean-reversion signals
        # Here we just reward mean-reversion features hypothetically
        reward += 5  # Reward for mean-reversion features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip the reward to ensure it remains within [-100, 100]
    return np.clip(reward, -100, 100)