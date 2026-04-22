import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]
    volatility_5d = s[135]
    volatility_20d = s[136]
    trend_r_squared = s[145]
    bb_position = s[149]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Using a multiple of average volatility
    low_volatility_threshold = avg_volatility / 2    # Lower threshold for caution
    
    # Initialize reward
    reward = 0.0
    
    # Reward structure
    if position == 0:  # Not holding position (BUY opportunities)
        if trend_r_squared > 0.8:  # Strong trend
            if bb_position < 0.2:  # Oversold condition
                reward += 50  # Strong buy opportunity
            elif volatility_5d < low_volatility_threshold:  # Low volatility
                reward += 30  # Buy with low volatility
            else:
                reward += 10  # Moderate buy opportunity
        elif bb_position < 0.2:  # Oversold condition without clear trend
            reward += 20  # Buy opportunity based on oversold condition

    elif position == 1:  # Holding position (SELL or HOLD opportunities)
        if trend_r_squared > 0.8:  # Strong trend
            reward += 10  # Encourage holding
        elif trend_r_squared < 0.5:  # Weak trend
            if bb_position > 0.8:  # Overbought condition
                reward -= 50  # Strong sell opportunity
            elif volatility_5d > high_volatility_threshold:  # High volatility
                reward -= 20  # Caution against holding
            else:
                reward += 5  # Continue holding with caution
        else:  # Mixed signals
            if bb_position > 0.8:  # Overbought condition
                reward -= 20  # Consider selling

    # Normalize the reward to be within the range of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward