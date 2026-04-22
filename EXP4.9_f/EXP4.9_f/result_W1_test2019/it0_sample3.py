import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Features list
    features = []
    
    # Calculate the last closing price
    last_close = s[114]  # closing price of day 19 (most recent)
    
    # Calculate daily returns (percentage change)
    daily_returns = [(s[i * 6 + 0] - s[i * 6 + 1]) / s[i * 6 + 1] if s[i * 6 + 1] != 0 else 0 for i in range(20)]
    
    # Feature 1: Average daily return over the past 20 days
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)
    
    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)
    features.append(volatility)
    
    # Feature 3: Price momentum (current close - close 5 days ago)
    price_momentum = last_close - s[114 - 5 * 6 + 0] if 114 - 5 * 6 >= 0 else 0
    features.append(price_momentum)
    
    # Optionally, we can add other features here
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Reward for upward features
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)