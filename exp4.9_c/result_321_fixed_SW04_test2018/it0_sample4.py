import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Change (percentage change from previous day)
    price_changes = np.zeros(20)
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:
            price_changes[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Volume Change (percentage change from previous day)
    volume_changes = np.zeros(20)
    for i in range(1, len(volumes)):
        if volumes[i-1] != 0:
            volume_changes[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    
    # Feature 3: Average True Range (ATR) 
    atr = np.zeros(20)
    for i in range(1, len(closing_prices)):
        high = s[i*6 + 2]  # High price of day i
        low = s[i*6 + 3]   # Low price of day i
        if i > 0:  # Ensure we have a previous day to calculate TR
            previous_close = closing_prices[i-1]
            tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
            atr[i] = tr
    
    # Calculate the average ATR over the last 14 days (if possible)
    atr_mean = np.mean(atr[max(0, len(atr) - 14):]) if len(atr) > 13 else np.nan
    
    features = [price_changes[-1], volume_changes[-1], atr_mean if not np.isnan(atr_mean) else 0]
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
        if len(features) > 0:
            reward += trend_direction * features[0] * 10.0  # Price change effect

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition for buying
            reward += 5.0  # Positive reward for buying in oversold condition
        elif features[0] > 0.01:  # Overbought condition for selling
            reward += 5.0  # Positive reward for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))