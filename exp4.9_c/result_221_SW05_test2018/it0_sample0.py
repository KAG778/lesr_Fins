import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate moving average, price change ratio and volume change
    closing_prices = s[0:120:6]  # Extracting the closing prices
    opening_prices = s[1:120:6]   # Extracting the opening prices
    volumes = s[4:120:6]          # Extracting the volumes
    
    # Feature 1: Moving Average of the last 5 days
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # If not enough data, fallback to last price
    
    # Feature 2: Price change ratio from the first to the last day
    price_change_ratio = (closing_prices[-1] - opening_prices[0]) / opening_prices[0] if opening_prices[0] != 0 else 0
    
    # Feature 3: Volume change (latest volume / mean volume over the last 5 days)
    if len(volumes) >= 5:
        average_volume = np.mean(volumes[-5:])
    else:
        average_volume = volumes[-1]  # If not enough data, fallback to last volume
    volume_change = volumes[-1] / average_volume if average_volume != 0 else 0
    
    features = [moving_average, price_change_ratio, volume_change]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Random moderate negative reward
        
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Favoring upward features
            reward += np.random.uniform(10, 20)  # Positive reward for trending up
        elif trend_direction < -0.3:
            # Favoring downward features
            reward += np.random.uniform(10, 20)  # Positive reward for trending down
            
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion
        # Penalize breakout-chasing features
        reward -= np.random.uniform(5, 15)  # Penalize for chasing breakouts
        
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]