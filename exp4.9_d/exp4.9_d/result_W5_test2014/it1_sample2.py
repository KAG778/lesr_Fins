import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Define relative thresholds
    high_volatility_threshold = 2 * volatility_20d  # High volatility threshold
    low_volatility_threshold = 0.5 * volatility_20d   # Low volatility threshold for potential buying
    
    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding
        # Strong BUY signal conditions
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4:  # Moderately strong trend
            reward += 30  # Moderate buy signal

        # Caution for buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Penalize for potential buy in high volatility
        elif volatility_20d < low_volatility_threshold:  # Low volatility environment
            reward += 10  # Reward for potential stability

    elif position == 1:  # Holding
        # Reward for holding in a strong trend
        if trend_r_squared > 0.8:
            reward += 30  # Strong hold signal
        elif trend_r_squared < 0.5:  # Trend weakening
            reward -= 50  # Strong penalty to encourage selling
        
        # Manage overbought conditions
        if bb_position > 0.8:  # Overbought condition
            reward -= 30  # Strong incentive to sell

        # Caution for holding in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Penalize for holding in high volatility

    # Normalize reward to be in the range of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward