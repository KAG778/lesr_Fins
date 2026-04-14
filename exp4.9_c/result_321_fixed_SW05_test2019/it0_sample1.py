import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, ..., 114)
    volumes = s[4::6]         # Extract trading volumes (indices 4, 10, ..., 114)
    
    features = []
    
    # Feature 1: Price Change Percentage
    price_change = np.zeros(len(closing_prices))
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:
            price_change[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    features.append(price_change[-1])  # Use the latest price change

    # Feature 2: 5-Day Simple Moving Average of Closing Prices
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # Fallback to last closing price
    features.append(moving_average)

    # Feature 3: Volume Change Percentage
    volume_change = np.zeros(len(volumes))
    for i in range(1, len(volumes)):
        if volumes[i-1] != 0:
            volume_change[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    features.append(volume_change[-1])  # Use the latest volume change

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
        if len(features) > 0:
            if trend_direction > 0:  # Uptrend
                reward += features[0] * 10.0  # Price change positive contribution
            else:  # Downtrend
                reward += -features[0] * 10.0  # Negative contribution for bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Oversold condition
            reward += 5.0  # Buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += -5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))