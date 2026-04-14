import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Compute features
    closing_prices = s[0::6]  # Extracting closing prices
    opening_prices = s[1::6]  # Extracting opening prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting volumes
    
    # Feature 1: Price Change (percentage change from the previous close)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume (over the last 20 days)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Normalized Price Range (high - low) / previous close
    price_range = (high_prices[-1] - low_prices[-1]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Return the new features as a numpy array
    return np.array([price_change, average_volume, price_range])

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
        # Consider SELL aligned features positively
        reward += 5.0 if features[0] < 0 else 0  # If price change is negative, reward for SELL
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Positive reward for upward price change
        else:  # Downtrend
            reward += -features[0] * 10.0  # Positive reward for downward price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold (price change negative)
            reward += 5.0  # Reward for BUY
        elif features[0] > 0:  # Overbought (price change positive)
            reward += 5.0  # Reward for SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))