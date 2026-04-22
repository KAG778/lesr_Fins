import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio
    
    # Initialize the reward
    reward = 0.0
    
    # Calculate average volatility for dynamic thresholds
    avg_volatility = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    # Buying phase (not holding)
    if position == 0:  
        # Strong buy signal
        if trend_r2 > 0.8 and bb_pos < 0.2:  
            reward += 50  # Strong buy opportunity
        elif trend_r2 > 0.6 and bb_pos < 0.4:  
            reward += 30  # Moderate buy opportunity
        elif vol_ratio > high_vol_threshold:  
            reward -= 20  # Caution in high volatility
        elif trend_r2 < 0.5:  
            reward -= 10  # Weak trend caution

    # Holding phase (currently holding)
    elif position == 1:  
        # Strong hold conditions
        if trend_r2 > 0.8:  
            reward += 30  # Strong encouragement to hold
        elif trend_r2 > 0.5 and bb_pos < 0.8:
            reward += 10  # Mild hold signal
        elif bb_pos > 0.8:  
            reward -= 40  # Strong encouragement to sell due to overbought
        elif trend_r2 < 0.5:  
            reward -= 30  # Encourage selling in weak trend
        if vol_ratio > high_vol_threshold:  
            reward -= 20  # Caution in high volatility

    # Ensure reward is normalized within the range [-100, 100]
    return np.clip(reward, -100, 100)