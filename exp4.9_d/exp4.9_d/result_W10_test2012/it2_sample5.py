import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Define thresholds
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    strong_trend_threshold = 0.8
    overbought_threshold = 0.8
    oversold_threshold = 0.2
    
    reward = 0
    
    if position == 0:  # Not holding
        # Strong buy signal
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4:
            reward += 30  # Reasonable buy opportunity
            
        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility
        # Encourage buying in low volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Reward cautious buying opportunity

    elif position == 1:  # Holding
        # Encourage holding in a strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 25  # Maintain position
        
        # Strong sell signal
        if bb_position > overbought_threshold or trend_r_squared < 0.5:
            reward -= 40  # Clear sell signal
        
        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against holding in high volatility
        # Mild reward for holding in stable conditions
        else:
            reward += 5  # Encourage stability

    # Normalize reward to ensure it's within the bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward