import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (holding or not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Calculate volatility threshold (multiples of 5-day volatility)
    volatility_threshold = 2 * volatility_5d
    
    # Initialize reward
    reward = 0
    
    # Reward structure based on position
    if position == 0:  # Not holding
        # Identify strong BUY opportunities
        if trend_r_squared > 0.8:  # Strong uptrend
            reward += 50  # Reward for potential BUY action
        if bb_position < 0.2:  # Oversold condition
            reward += 30  # Additional reward for oversold condition
        if volatility_20d > volatility_threshold:  # Caution in high volatility
            reward -= 20  # Penalty for buying in high volatility
        
    elif position == 1:  # Holding
        # Reward for holding during uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Reward for holding in a strong trend
        if bb_position > 0.8:  # Overbought condition
            reward -= 40  # Penalty for not selling in overbought condition
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Encourage to sell when trend weakens

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward