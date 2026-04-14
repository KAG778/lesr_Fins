import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Ensure we have 20 days of data
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes
    
    # Feature 1: Percentage Change in Closing Price
    price_change = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_change[i-1] = 0  # If previous price is 0, change is 0
    features.append(np.mean(price_change))  # Average price change over the period
    
    # Feature 2: Moving Average (5-day MA of closing prices)
    if len(closing_prices) >= 5:
        ma_5 = np.mean(closing_prices[-5:])
    else:
        ma_5 = closing_prices[-1]  # If not enough data, use the last closing price
    features.append(ma_5)
    
    # Feature 3: Volume Change (% Change in Volume)
    volume_change = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_change[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]
        else:
            volume_change[i-1] = 0  # If previous volume is 0, change is 0
    features.append(np.mean(volume_change))  # Average volume change over the period
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
    
    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for correct upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for correct downward features
    
    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Mild positive reward for mean-reversion features
    
    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)