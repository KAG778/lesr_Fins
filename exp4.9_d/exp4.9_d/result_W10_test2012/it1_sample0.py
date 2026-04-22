import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Define thresholds for volatility
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Conditions for buying
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_position < 0.3:  # Moderate trend and moderate oversold
            reward += 30  # Reasonable buy signal
        
        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encouragement in low volatility

    else:  # Holding
        # Conditions for holding
        if trend_r_squared > 0.8:  # Strong trend
            reward += 25  # Encourage holding
        
        # Conditions for selling
        if bb_position > 0.8:  # Overbought condition
            reward += 40  # Strong sell signal
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Strong sell signal

        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution advised, potential for sell

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward