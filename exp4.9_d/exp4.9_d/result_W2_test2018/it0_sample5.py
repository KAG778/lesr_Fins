import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Constants
    MAX_REWARD = 100
    MIN_REWARD = -100
    
    # Position flag
    position = s[150]
    
    # Historical Volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    vol_mean = (vol_5d + vol_20d) / 2
    vol_threshold = vol_mean * 2  # Define a threshold based on historical volatility
    
    # Trend features
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend, oversold
            reward += MAX_REWARD * 0.5  # Strong BUY signal
        elif bb_pos > 0.8:  # Overbought condition
            reward += MIN_REWARD * 0.2  # Mild penalty for buying overbought

    elif position == 1:  # Holding
        if trend_r2 > 0.8:  # Strong trend, encourage holding
            reward += MAX_REWARD * 0.3  # Positive reward for holding
        elif trend_r2 < 0.5:  # Weak trend, consider selling
            reward += MIN_REWARD * 0.5  # Penalty for holding in weak trend
        elif bb_pos > 0.8:  # Overbought, consider selling
            reward += MIN_REWARD * 0.7  # Strong penalty for holding overbought
            
    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, MIN_REWARD, MAX_REWARD)
    
    return reward