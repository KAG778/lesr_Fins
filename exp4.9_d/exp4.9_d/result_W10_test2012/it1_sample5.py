import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Conditions for buying
        if trend_r2 > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_position < 0.3:  # Moderate trend and reasonable buy
            reward += 30
        elif volatility_5d < low_volatility_threshold:  # Low volatility environment
            reward += 10  # Cautious buying opportunity
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Discourage buying in high volatility

    elif position == 1:  # Holding
        # Conditions for holding
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Encourage holding
        elif trend_r2 < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 50  # Strong sell signal
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 30  # Caution advised, potential for sell
        else:
            reward += 5  # Mild reward for holding in stable conditions
    
    # Ensure the reward is within the bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward