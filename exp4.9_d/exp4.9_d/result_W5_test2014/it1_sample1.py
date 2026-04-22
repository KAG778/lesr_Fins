import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define thresholds
    high_volatility_threshold = 2 * volatility_20d  # High volatility threshold
    low_volatility_threshold = 0.5 * volatility_20d  # Low volatility threshold
    trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition for BB position
    oversold_threshold = 0.2  # Oversold condition for BB position
    
    # Initialize reward
    reward = 0.0
    
    if position == 0:  # Not holding the stock
        # Identify strong BUY opportunities
        if trend_r_squared > trend_threshold and bb_position < oversold_threshold:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif bb_position < oversold_threshold:  # Considered oversold
            reward += 30  # Moderate buy signal
            
        # Penalize for potential buy in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
        
    elif position == 1:  # Holding the stock
        # Reward for holding in strong trends
        if trend_r_squared > trend_threshold:  # Confirmed uptrend
            reward += 30  # Strong hold
        elif trend_r_squared < 0.5 or bb_position > overbought_threshold:  # Weak trend or overbought
            reward -= 50  # Signal to consider selling
            
        # Penalize for holding in high volatility
        if volatility_20d > high_volatility_threshold: 
            reward -= 15  # Caution in high volatility

    # Normalize the reward to be in the range of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward