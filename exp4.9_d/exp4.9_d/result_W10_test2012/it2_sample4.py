import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Buying conditions
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4:  # Moderate uptrend
            reward += 30  # Reasonable buy opportunity
        
        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against buying in high volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encourage buying in low volatility

    elif position == 1:  # Holding
        # Conditions for holding
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Encourage holding
        
        # Selling conditions
        if bb_position > 0.8:  # Overbought condition
            reward += 40  # Strong sell signal
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Strong signal to consider selling

        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution advised for holding in high volatility

    # Ensure the reward stays within bounds [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward