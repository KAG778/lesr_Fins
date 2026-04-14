import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate the features based on the last 20 days of trading data
    closing_prices = s[0:120:6]  # Get closing prices (s[0], s[6], ..., s[114])
    high_prices = s[3:120:6]      # Get high prices (s[3], s[9], ..., s[117])
    low_prices = s[4:120:6]       # Get low prices (s[4], s[10], ..., s[118])
    volumes = s[4:120:6]          # Get volumes (s[4], s[10], ..., s[118])

    # Feature 1: Price Momentum (percentage change in closing price)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Trading Volume
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Price Range (High - Low)
    price_range = np.mean(high_prices - low_prices) if len(high_prices) > 0 else 0.0

    features = [price_momentum, avg_volume, price_range]
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
        reward -= 40.0  # Strong negative for BUY-aligned actions
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price momentum
            reward += trend_direction * 10.0  # Reward for following the trend
        elif features[0] < 0:  # Negative momentum, penalize if trends are against it
            reward -= trend_direction * 10.0

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Buy signal
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))