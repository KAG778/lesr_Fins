import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate price change (current closing price - previous closing price) / previous closing price
    # to avoid division by zero, we will check if the previous closing price is non-zero
    closing_prices = s[0::6]
    price_change = np.zeros(20)
    for i in range(1, 20):
        if closing_prices[i-1] != 0:
            price_change[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_change[i] = 0.0  # Handle edge case
    
    features.append(np.mean(price_change[1:]))  # mean price change excluding the first day
    
    # Calculate average volume over the last 20 days
    trading_volumes = s[4::6]
    average_volume = np.mean(trading_volumes)
    features.append(average_volume)
    
    # Calculate RSI for the last 14 days (standard calculation)
    # Using closing prices to calculate RSI
    if len(closing_prices) < 14:
        rsi = 0.0  # Handle edge case when we don't have enough data
    else:
        gains = np.where(price_change[1:] > 0, price_change[1:], 0)
        losses = np.where(price_change[1:] < 0, -price_change[1:], 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        # Avoid division by zero
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    
    features.append(rsi)
    
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
        reward += np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY-aligned features
        # Assume SELL-aligned features are present in the context, we would reward them positively
        reward += np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(10, 20)  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Ensure reward is within the [-100, 100] range