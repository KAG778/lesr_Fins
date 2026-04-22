import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    trend_strength = s[145]  # R² of trend
    bb_position = s[149]  # Bollinger Band position
    volatility_5d = s[135]  # 5-day volatility
    volatility_20d = s[136]  # 20-day volatility
    avg_volatility = np.mean([volatility_5d, volatility_20d])
    
    # Define thresholds
    high_volatility_threshold = 2 * avg_volatility
    overbought_threshold = 0.8
    oversold_threshold = 0.2
    strong_trend_threshold = 0.8
    
    reward = 0.0
    
    # Reward logic when NOT holding (position = 0)
    if position == 0.0:
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > strong_trend_threshold and bb_position < 0.4: 
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:
            reward -= 20  # Discourage buying in high volatility
    
    # Reward logic when holding (position = 1)
    else:
        if trend_strength > strong_trend_threshold:
            reward += 20  # Encourage holding in a strong trend
        elif trend_strength < 0.5 or bb_position > overbought_threshold:
            reward -= 40  # Strong sell signal
        elif bb_position > overbought_threshold and trend_strength < strong_trend_threshold:
            reward -= 30  # Consider selling in overbought conditions
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution advised for holding in high volatility

    # Ensure reward is within bounds [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward