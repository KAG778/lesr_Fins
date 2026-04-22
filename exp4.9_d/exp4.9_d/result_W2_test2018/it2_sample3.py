import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features
    s = enhanced_state
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    price_pos = s[146]  # Price position in the 20-day range [0, 1]
    
    # Define relative volatility thresholds
    high_volatility_threshold = 2 * np.mean([volatility_5d, volatility_20d])
    low_volatility_threshold = 0.5 * np.mean([volatility_5d, volatility_20d])
    
    # Initialize reward
    reward = 0
    
    if position == 0:  # Not holding the stock
        # Strong buy signal: trending up and oversold
        if trend_r_squared > 0.8 and bb_pos < 0.2 and volatility_20d < low_volatility_threshold:
            reward += 50  # Strong buy opportunity
        
        # Moderate buy signal: trending moderately and near oversold
        elif trend_r_squared > 0.6 and bb_pos < 0.3:
            reward += 30  # Moderate buy opportunity
        
        # Caution against buying in high volatility
        elif volatility_20d > high_volatility_threshold:
            reward -= 20  # Penalize buying in high volatility

    elif position == 1:  # Holding the stock
        # Reward for holding during a strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Encourage holding
        
        # Consider selling if trend weakens or overbought
        elif trend_r_squared < 0.5 or bb_pos > 0.8:
            reward -= 30  # Strong sell signal
        
        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility markets
            
    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)