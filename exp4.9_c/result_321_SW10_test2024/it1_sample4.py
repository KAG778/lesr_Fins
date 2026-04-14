import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract trading volumes
    
    # Feature 1: Price Momentum (current closing price vs opening price)
    price_momentum = (closing_prices[-1] - opening_prices[-1]) / (opening_prices[-1] + 1e-10)

    # Feature 2: Average True Range (ATR) over the last 20 days
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 0.0

    # Feature 3: Volume Change (% Change from previous volume)
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-10) if len(volumes) > 1 else 0

    # Feature 4: Price Range normalized by closing price
    price_range = (high_prices[-1] - low_prices[-1]) / (closing_prices[-1] + 1e-10)

    return np.array([price_momentum, atr, volume_change, price_range])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0  # Strong negative for BUY-aligned features
        reward += 10.0 * (1 - features[2])  # MILD positive for SELL-aligned features based on volume change
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 20.0  # Reward for following trend based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif features[0] > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))