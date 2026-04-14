import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    high_prices = s[2:120:6]      # Extracting high prices
    low_prices = s[3:120:6]       # Extracting low prices
    volumes = s[4:120:6]          # Extracting volumes

    # Feature 1: Price Momentum (current close - close from 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if len(closing_prices) >= 6 and closing_prices[5] != 0 else 0.0
    
    # Feature 2: Volume Change (percentage change from the last day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] if len(volumes) > 1 and volumes[1] != 0 else 0.0
    
    # Feature 3: Average True Range (ATR) over the last 14 days
    atr_values = []
    for i in range(1, len(closing_prices)):
        high = high_prices[i]
        low = low_prices[i]
        previous_close = closing_prices[i-1]
        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        atr_values.append(tr)

    atr_mean = np.mean(atr_values[-14:]) if len(atr_values) >= 14 else 0.0

    # Feature 4: Price Range (high - low of the last day)
    price_range = high_prices[0] - low_prices[0] if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    features = [price_momentum, volume_change, atr_mean, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:
            reward += 5.0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10.0 if features[0] > 0 else 0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Align rewards with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Positive for buying
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))