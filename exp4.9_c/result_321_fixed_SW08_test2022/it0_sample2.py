import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Compute features based on OHLCV data
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price Change (percentage change from the previous closing price)
    price_change = np.zeros(19)  # 19 days, since we have 20 days of data
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:
            price_change[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]

    # Feature 2: Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices)

    # Feature 3: Average Volume (average trading volume over the last 20 days)
    average_volume = np.mean(volumes)

    # Return the computed features as a numpy array
    return np.array([price_change[-1], volatility, average_volume])  # Last price change, volatility, average volume

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
        reward -= 40.0  # Strong negative for buying
        # Reward for selling (if features indicate a sell signal):
        reward += 5.0 if features[0] < 0 else 0  # Mild positive if recent price change is negative
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Positive reward for upward price change
        elif trend_direction < 0:  # Downtrend
            reward += -features[0] * 10.0  # Positive reward for downward price change (correct bearish bet)

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 10.0  # Mild positive for buying
        elif features[0] > 0.01:  # Overbought condition
            reward += 10.0  # Mild positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))