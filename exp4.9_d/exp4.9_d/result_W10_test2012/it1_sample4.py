import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Relative threshold for caution

    reward = 0
    
    # If not holding (position = 0)
    if position == 0:  
        # Buy signal conditions
        if trend_strength > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_strength > 0.6 and bb_position < 0.4:  # Moderate buy opportunity
            reward += 30  # Moderate buy signal
        
        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against buying in volatile conditions
    
    # If holding (position = 1)
    elif position == 1:  
        # Conditions for holding
        if trend_strength > 0.8:  # Strong uptrend
            reward += 20  # Encourage holding
        
        # Conditions for selling
        if bb_position > 0.8:  # Overbought condition
            reward += 30  # Strong signal to sell
        elif trend_strength < 0.5:  # Weak trend
            reward -= 30  # Strong signal to consider selling
        
        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Caution against holding in volatile conditions
    
    # Ensure the reward stays within bounds
    reward = np.clip(reward, -100, 100)
    
    return reward