import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Relative thresholds based on volatility
    high_volatility_threshold = 1.5 * volatility_20d  # Adjusted for caution
    low_volatility_threshold = 0.5 * volatility_20d  # Oversold condition
    
    # Initialize reward
    reward = 0.0

    # Differentiating actions based on position
    if position == 0:  # Not holding
        # Strong uptrend and oversold condition
        if trend_r_squared > 0.8 and bb_position < 0.2:
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_position < 0.3:
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:
            reward -= 15  # Penalty for potential buy in high volatility
        elif bb_position > 0.5:  # Neutral zone
            reward -= 10  # Slight penalty for uncertainty

    elif position == 1:  # Holding
        # Reward for holding in strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Reward for holding in a strong trend
        elif trend_r_squared < 0.5:  # Weakening trend
            reward -= 40  # Strong signal to consider selling
        elif bb_position > 0.8:  # Overbought condition
            reward -= 30  # Penalty for not selling in overbought
        elif volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
    
    # Normalize the reward to be in the range of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward