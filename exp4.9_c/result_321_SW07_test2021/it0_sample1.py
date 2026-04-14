import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate moving averages
    closing_prices = s[0::6]  # Closing prices for the last 20 days
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    moving_average_10 = np.mean(closing_prices[-10:]) if len(closing_prices[-10:]) > 0 else 0
    features.append(moving_average_5)
    features.append(moving_average_10)
    
    # Calculate Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    # Calculate price volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)
    
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
        if enhanced_state[123] > 0:  # Assuming positive features align with BUY
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
        return reward
    
    if risk_level > 0.4:
        reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals
        return reward
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Example features -> assuming 0 for oversold and 1 for overbought
        if enhanced_state[123] < 0:  # Assuming negative features are mean-reversion aligned
            reward += 10  # Reward for mean-reversion features
        else:
            reward -= 10  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)
    
    return np.clip(reward, -100, 100)  # Ensure reward stays in [-100, 100]