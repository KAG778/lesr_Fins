import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Position: 1.0 if holding, 0.0 if not
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # R² of trend
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 1
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    high_volatility_threshold = 1.5  # 1.5 times the 20-day volatility for caution
    strong_trend_threshold = 0.8  # Threshold for strong trend
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition
    
    if position == 0:  # Not holding
        # Strong buy signal conditions
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > strong_trend_threshold and volatility_ratio < high_volatility_threshold:
            reward += 30  # Moderate buy opportunity
        else:
            reward -= 10  # Neutral or weak buy conditions
        
    else:  # Holding position
        # Reward for holding during a strong trend
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong trend
        # Selling conditions
        if bb_position > overbought_threshold or trend_r_squared < 0.5:
            reward -= 40  # Strong sell signal due to overbought or weak trend
        elif volatility_ratio > high_volatility_threshold:
            reward -= 20  # Be cautious in high volatility
        
    # Penalize for frequent trading by adding a small negative reward for neutral signals
    if reward == 0:
        reward -= 5  # Discourage inaction in uncertain conditions

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward