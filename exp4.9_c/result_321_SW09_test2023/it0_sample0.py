import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (days 0 to 19)
    volumes = s[4:120:6]          # Extract trading volumes (days 0 to 19)
    
    # Calculate features
    price_momentum = closing_prices[19] - closing_prices[18]  # Difference between last two closing prices
    price_momentum_feature = price_momentum / closing_prices[18] if closing_prices[18] != 0 else 0
    
    volume_change = (volumes[19] - volumes[18]) / volumes[18] if volumes[18] != 0 else 0  # Percentage change in volume
    
    # Calculate volatility as standard deviation of closing prices
    volatility_measure = np.std(closing_prices)

    features = [price_momentum_feature, volume_change, volatility_measure]
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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price momentum
            reward += trend_direction * 10.0 * features[0]  # Reward for positive trend
        elif features[0] < 0:  # Negative price momentum
            reward += trend_direction * -5.0 * features[0]  # Penalize for incorrect trend

    # Priority 3: Sideways (Mean Reversion)
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Reward for potential buy signal
        elif features[0] > 0:  # Overbought condition
            reward += -5.0  # Penalize for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))