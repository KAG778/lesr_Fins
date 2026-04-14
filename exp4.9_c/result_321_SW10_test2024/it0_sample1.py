import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Change Percentage
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    feature1 = price_change

    # Feature 2: Average Volume over last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    feature2 = avg_volume

    # Feature 3: High-Low Price Range
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0
    feature3 = price_range

    return np.array([feature1, feature2, feature3])

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
        # Strong negative reward for BUY-aligned features
        reward -= 40.0 * features[0]  # Price change feature impacts reward
        reward += 5.0  # MILD positive for SELL-aligned features
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0 * features[0]  # Price change feature impacts reward

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10.0 * features[0]  # Buy signal aligned with uptrend (price change)
        else:  # Downtrend
            reward += 10.0 * -features[0]  # Sell signal aligned with downtrend (price change)

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Encourage BUY
        elif features[0] > 0.01:  # Overbought condition
            reward += 5.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))