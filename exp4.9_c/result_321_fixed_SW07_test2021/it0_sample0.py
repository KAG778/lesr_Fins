import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract the closing prices (every 6th element starting from index 0)
    volumes = s[4::6]         # Extract the volumes (every 6th element starting from index 4)
    
    # Feature 1: Price Change (percentage)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Volume Change (percentage)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0

    # Feature 3: 5-day Moving Average of the last closing prices
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # If less than 5 days, use the last closing price as MA
    
    # Return the computed features as a numpy array
    features = [price_change, volume_change, moving_average]
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
        reward -= 40.0  # Strong negative for BUY
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Price change is positive
            reward += 10.0 * features[0]  # Strong reward for upward movement
        elif features[0] < 0:  # Price change is negative
            reward += 10.0 * features[0]  # Reward for downward movement if trend is bearish

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buying opportunity
        elif features[0] > 0.01:  # Overbought condition
            reward += 5.0  # Reward for potential selling opportunity

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))