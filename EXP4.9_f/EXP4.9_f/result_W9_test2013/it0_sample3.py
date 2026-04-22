import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    # Price Change Feature
    price_change = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:
            price_change[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Volume Change Feature
    volume_change = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if trading_volumes[i-1] != 0:
            volume_change[i-1] = (trading_volumes[i] - trading_volumes[i-1]) / trading_volumes[i-1]
    
    # Simple Moving Average (SMA) of closing prices for last 5 days
    sma = np.zeros(20)
    for i in range(4, 20):
        sma[i] = np.mean(closing_prices[i-4:i+1])
    
    # Compile features
    features = []
    features.extend(price_change)  # 19 features for price change
    features.extend(volume_change)  # 19 features for volume change
    features.append(sma[-1])  # 1 feature for the latest SMA
    
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
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)  # Moderate negative reward
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Trend following positive reward
        if trend_direction > 0:
            reward += np.random.uniform(5, 15)  # Positive reward for upward features
        else:
            reward += np.random.uniform(5, 15)  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Mean-reversion features
        # Assuming features after index 123 include indicators for oversold/overbought
        # Placeholder logic, implement specific conditions based on actual features
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range