import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (day i is at index 6*i)
    volumes = s[4:120:6]          # Extract volumes (day i is at index 6*i + 4)

    # Feature 1: Price Momentum (latest closing price - closing price 4 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-5] if len(closing_prices) > 4 else 0

    # Feature 2: Volume Change (percentage change from last day to the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Feature 3: Volatility (standard deviation of closing prices over the last 5 days)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0

    # Return the computed features as a numpy array
    return np.array([price_momentum, volume_change, volatility])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY (action 0)
        reward -= 50 if features[0] > 0 else 0  # Assuming features[0] relates to buying signals
        reward += 10 if features[1] < 0 else 0  # Assuming features[1] relates to selling signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20 if features[0] > 0 else 0

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10 if features[0] > 0 else 0  # Positive for upward features
        elif trend_direction < -0.3:
            reward += 10 if features[0] < 0 else 0  # Positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10 if features[0] < 0 else 0  # Oversold situation (buy)
        reward -= 10 if features[0] > 0 else 0  # Overbought situation (sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)