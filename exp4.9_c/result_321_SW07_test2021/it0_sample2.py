import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting volumes
    
    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[19] - closing_prices[14]  # Day 19 - Day 14
    # Feature 2: Average Volume over the last 5 days
    average_volume = np.mean(volumes[-5:]) if np.any(volumes[-5:]) else 0  # Avoid division by zero
    # Feature 3: Price RSI (14-period RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    features = [price_momentum, average_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # If price momentum is positive
            reward = -50  # Strong negative for BUY
        else:
            reward = 10  # MILD POSITIVE for SELL
    elif risk_level > 0.4:
        reward = -20 if features[0] > 0 else 0  # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Oversold condition for RSI
            reward += 10  # Encourage buying
        elif features[2] > 70:  # Overbought condition for RSI
            reward += 10  # Encourage selling
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward by 50%
    
    return float(reward)