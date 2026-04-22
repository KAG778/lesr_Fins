import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # High volatility ratio
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB position
    overbought_threshold = 0.8  # Overbought condition for BB position
    
    # Differentiate actions based on position
    if position == 0:  # Not holding the stock
        # Strong buy signal: strong trend and oversold
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal: low volatility and weak trend
        elif volatility_ratio < 1.5 and trend_r_squared < 0.5:
            reward += 20  # Moderate buy opportunity
        else:
            reward -= 10  # Neutral or caution signal

    else:  # Holding the stock
        # Encourage holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Reward for holding in a strong trend
        # Penalize for overbought conditions
        if bb_position > overbought_threshold:
            reward -= 50  # Negative reward for holding overbought
        # Penalize for weak trend
        if trend_r_squared < 0.5:
            reward -= 20  # Consider selling as trend weakens

    # Penalize for extreme volatility
    if volatility_ratio > high_volatility_threshold:
        reward -= 30  # Caution in extreme volatility regime

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward