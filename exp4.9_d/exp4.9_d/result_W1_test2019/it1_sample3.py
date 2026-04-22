import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Caution for high volatility
    low_volatility_threshold = avg_volatility / 2    # Caution for low volatility

    # Trading logic based on position
    if position == 0:  # Not holding
        # Strong buying conditions
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong trend and undervalued
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_pos < 0.4:  # Moderate buy signal
            reward += 30  # Buy opportunity
        elif trend_r_squared < 0.5 and bb_pos < 0.3:  # Weak trend but low BB position
            reward += 10  # Cautious buy signal
        
        # Penalty for buying in extreme volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution advised in high volatility

    elif position == 1:  # Holding
        # Encourage holding in strong trends
        if trend_r_squared > 0.8:  # Strong uptrend
            reward += 20  # Positive reward for holding
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Negative reward for holding in a weak trend
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong sell opportunity
        
        # Penalty for holding in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution against holding
        
        # Reward for holding in low volatility
        if volatility_20d < low_volatility_threshold:
            reward += 10  # Reward for holding in stable conditions

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward