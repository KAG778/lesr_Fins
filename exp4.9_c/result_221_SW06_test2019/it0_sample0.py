import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    # Extracting the last 20 days of closing prices and volumes
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]

    # Feature 1: Price Momentum (change from the last day)
    price_momentum = closing_prices[-1] / closing_prices[-2] - 1 if closing_prices[-2] != 0 else 0
    
    # Feature 2: Average Trading Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Price Range (high - low) over the last day
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0

    # Return the features as a numpy array
    return np.array([price_momentum, average_volume, price_range])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for buying in high-risk environment
        if features[0] > 0:  # Assuming feature[0] indicates a BUY signal
            reward = np.random.uniform(-50, -30)
        else:  # Assuming feature indicates a SELL signal
            reward = np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = -10

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # BUY aligned with an uptrend
            reward = 10
        elif trend_direction < -0.3 and features[0] < 0:  # SELL aligned with a downtrend
            reward = 10

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition, BUY signal
            reward = 10
        elif features[0] > 0:  # Overbought condition, SELL signal
            reward = -10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)