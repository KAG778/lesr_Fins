import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Closing prices for the last 20 days
    closing_prices = s[0::6]
    
    # Feature 1: Price Momentum (Rate of Change)
    # Calculate the momentum as the percentage change from the closing price 19 days ago
    if closing_prices[0] != 0:
        momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19]
    else:
        momentum = 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change over the last 20 days
    volumes = s[4::6]  # Extract trading volumes
    average_volume = np.mean(volumes)
    last_volume = volumes[0]
    if average_volume != 0:
        volume_change = (last_volume - average_volume) / average_volume
    else:
        volume_change = 0
    features.append(volume_change)
    
    # Feature 3: Bollinger Band Width (Volatility measure)
    # Calculate the standard deviation of closing prices
    std_dev = np.std(closing_prices)
    if np.mean(closing_prices) != 0:
        bollinger_band_width = std_dev / np.mean(closing_prices)
    else:
        bollinger_band_width = 0
    features.append(bollinger_band_width)
    
    return np.array(features)

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
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # Assuming momentum is a BUY signal
            reward -= np.random.uniform(30, 50)
        # MILD POSITIVE reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative momentum is a SELL signal
            reward += np.random.uniform(5, 10)
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming momentum is a BUY signal
            reward -= np.random.uniform(10, 20)
    
    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += np.random.uniform(10, 20)
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0:  # Overbought condition
            reward -= np.random.uniform(10, 20)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds