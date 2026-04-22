import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]

    # Calculate relative volatility thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding (buy signals)
        # Encourage buying on strong trends and oversold conditions
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong buy condition
            reward += 50
        elif trend_r_squared > 0.7 and bb_position < 0.3:  # Moderate buy condition
            reward += 30
            
        # Penalize for buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encourage buying in low volatility

    elif position == 1:  # Holding (sell signals)
        # Encourage holding during strong trends
        if trend_r_squared > 0.8:
            reward += 20  # Maintain position
        
        # Conditions for selling
        if bb_position > 0.8:  # Overbought condition
            reward += 40  # Strong sell signal
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Strong signal to consider selling
            
        # Penalize for holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution advised for holding in high volatility

    # Normalize reward to ensure it stays within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward