import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    
    # Calculate features
    # Feature 1: Price Momentum (C[n] - C[n-1])
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    
    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Volatility (Standard deviation of closing prices)
    volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0.0
    
    # Return computed features as a numpy array
    return np.array([price_momentum, average_volume, volatility])

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= 40.0  # Adjust as needed
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price momentum
            reward += features[0] * 10.0  # Adjust multiplier as needed
        elif features[0] < 0:  # Negative price momentum
            reward += -features[0] * 10.0  # Penalize if betting against trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold
            reward += 5.0  # Encourage buying
        elif features[0] > 0:  # Overbought
            reward += -5.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Return reward value clipped to the range [-100, 100]
    return float(np.clip(reward, -100, 100))