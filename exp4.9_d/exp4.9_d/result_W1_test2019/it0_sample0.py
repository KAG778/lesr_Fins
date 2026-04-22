import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Retrieve relevant features
    position = s[150]
    volatility_5d = s[135]
    volatility_20d = s[136]
    trend_r_squared = s[145]
    bb_position = s[149]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    if position == 0:  # Not holding
        # Buying conditions
        if trend_r_squared > 0.8:  # Strong trend
            if s[128] < 30:  # RSI for oversold condition
                reward += 50  # Strong buy signal
            elif bb_position < 0.2:  # Low BB position (undervalued)
                reward += 30  # Buy opportunity
        if volatility_20d > high_vol_threshold:  # Extreme volatility
            reward -= 10  # Caution advised, discourage aggressive buys
        
    elif position == 1:  # Holding
        # Holding conditions
        if trend_r_squared > 0.8:  # Strong uptrend
            reward += 10  # Encouragement to hold
        elif bb_position > 0.8:  # Overbought condition
            reward -= 20  # Consider selling
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Time to reconsider position
    
    # Normalize reward to be between -100 and 100
    reward = np.clip(reward, -100, 100)
    
    return reward