import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract every 6th element starting from index 0 (closing prices)
    volumes = s[4::6]  # Extract every 6th element starting from index 4 (volumes)

    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change_pct[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]

    # Feature 2: Moving Average (last 5 days)
    moving_average = np.zeros(20)
    for i in range(4, 20):
        moving_average[i] = np.mean(closing_prices[i-4:i+1])  # Simple moving average

    # Feature 3: Volume Change Percentage
    volume_change_pct = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_change_pct[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]

    # Combine features and return as a numpy array
    features = np.concatenate((price_change_pct, moving_average, volume_change_pct))
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
        reward -= 45.0  # Strong negative for risky buy
    elif risk_level > 0.4:
        reward -= 15.0  # Moderate negative for risky buy

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price change percentage
            reward += 10.0 * trend_direction  # Favorable to the trend
        elif features[0] < 0:  # Negative price change percentage
            reward += -10.0 * trend_direction  # Unfavorable to the trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.02:  # Oversold condition
            reward += 10.0  # Encourage buying
        elif features[0] > 0.02:  # Overbought condition
            reward += 5.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))