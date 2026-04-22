import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate volatility thresholds
    volatility_threshold = 2 * volatility_5d  # Example of using 2x the 5-day volatility
    
    # Initialize reward
    reward = 0
    
    # Reward mechanics based on position
    if position == 0:  # Not holding
        # Consider buy opportunities
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Base reward for strong trend
        if bb_pos < 0.2:  # Oversold condition
            reward += 30  # Strong buy signal
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 10  # Discouraging buying in overbought conditions
        
    elif position == 1:  # Holding
        if trend_r_squared > 0.8:  # Strong trend
            reward += 15  # Reward for holding in a strong trend
        if bb_pos > 0.8:  # Overbought condition
            reward -= 20  # Consider selling
        if volatility_20d > volatility_threshold:  # High volatility environment
            reward -= 15  # Caution in high volatility
            
        # Check for trend weakening (could be more complex based on other indicators)
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 25  # Encourage selling in weak trends
    
    # Normalize reward to a range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward