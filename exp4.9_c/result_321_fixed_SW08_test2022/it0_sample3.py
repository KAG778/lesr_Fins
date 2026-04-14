import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4:120:6]          # Extract volumes (every 6th element starting from index 4)
    
    # Feature 1: Price Momentum (current closing price - previous closing price)
    price_momentum = closing_prices[0] - closing_prices[1] if len(closing_prices) > 1 else 0
    
    # Feature 2: Average Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Price Range (high - low)
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0
    
    features = [price_momentum, average_volume, price_range]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # If price momentum is positive
            reward += 10.0  # Positive reward for bullish trend
        elif features[0] < 0:  # If price momentum is negative
            reward += 10.0  # Positive reward for bearish trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.5:  # Oversold condition
            reward += 5.0  # Buy signal
        elif features[0] > 0.5:  # Overbought condition
            reward += 5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))