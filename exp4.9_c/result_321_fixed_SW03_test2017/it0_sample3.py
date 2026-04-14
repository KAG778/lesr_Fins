import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state (OHLCV for 20 days)
    
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]  # Extract trading volumes

    # Feature 1: Price Momentum (current closing price vs. closing price 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / (closing_prices[5] if closing_prices[5] != 0 else 1)

    # Feature 2: Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices)

    # Feature 3: Volume Change (percentage change from average volume)
    average_volume = np.mean(volumes)
    volume_change = (volumes[0] - average_volume) / (average_volume if average_volume != 0 else 1)

    # Return the computed features as a numpy array
    return np.array([price_momentum, volatility, volume_change])

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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for buying
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:  # Check if features contain data
            reward += trend_direction * features[0] * 10.0  # Price momentum based reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 5.0  # Mild positive for mean-reversion buy signal
        elif features[0] > 0.1:  # Overbought condition
            reward += 5.0  # Mild positive for mean-reversion sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))