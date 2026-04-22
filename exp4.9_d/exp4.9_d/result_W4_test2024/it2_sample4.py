import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio
    
    # Calculate dynamic thresholds based on historical volatility
    avg_volatility = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    # Initialize the reward
    reward = 0.0
    
    if position == 0:  # Not holding stock (BUY phase)
        # Strong buy opportunity
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy signal
            
        # Risk management for high volatility
        if vol_ratio > high_vol_threshold:  
            reward -= 20  # Caution in high volatility markets
        elif vol_ratio < low_vol_threshold:  
            reward += 10  # Encouragement in low volatility environments

    else:  # Holding stock (SELL/HOLD phase)
        # Encourage holding during a strong trend
        if trend_r2 > 0.8:  
            reward += 30  # Positive reward for holding
        elif trend_r2 > 0.5 and bb_pos < 0.8:  
            reward += 10  # Mild hold signal
        
        # Selling conditions
        if bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong encouragement to sell
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Strong encouragement to sell
        
        # Caution in high volatility
        if vol_ratio > high_vol_threshold:  
            reward -= 30  # Penalize for holding in high volatility

    # Normalize the reward to ensure it stays within bounds
    reward = np.clip(reward, -100, 100)
    
    return reward