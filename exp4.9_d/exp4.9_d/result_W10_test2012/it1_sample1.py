import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    high_volatility_threshold = 2 * np.mean([volatility_5d, volatility_20d])
    low_volatility_threshold = 0.5 * np.mean([volatility_5d, volatility_20d])
    
    # Initialize reward
    reward = 0.0

    if position == 0.0:  # Not holding
        # Encourage buying on strong trend and oversold condition
        if trend_r2 > 0.8 and bb_pos < 0.2:
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:
            reward += 30  # Moderate buy signal
        # Penalize for buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    elif position == 1.0:  # Holding
        # Encourage holding in a strong uptrend
        if trend_r2 > 0.8:
            reward += 20  # Maintain position
        # Penalize for selling in high volatility or weak trend
        if trend_r2 < 0.6 or bb_pos > 0.8:
            reward -= 40  # Clear signal to sell
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against holding in high volatility

    # Ensure the reward stays within bounds
    reward = np.clip(reward, -100, 100)
    
    return reward