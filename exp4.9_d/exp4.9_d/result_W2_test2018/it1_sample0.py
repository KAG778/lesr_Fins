import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features
    position = s[150]  # 1.0 = holding stock; 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position
    price_pos = s[146]  # Price position in 20-day range [0, 1]
    
    # Define relative thresholds based on volatility
    volatility_high_threshold = 2 * volatility_20d
    volatility_low_threshold = 0.5 * volatility_20d
    
    # Initialize reward
    reward = 0
    
    if position == 0:  # Not holding
        # Strong buy signal
        if trend_r_squared > 0.8 and price_pos < 0.2:  # Strong trend, oversold
            reward += 50  # Strong buy opportunity
        
        # Moderate buy signal
        elif trend_r_squared > 0.5 and price_pos < 0.4:  # Moderate trend, near oversold
            reward += 30  # Moderate buy opportunity
        
        # Caution against high volatility
        elif volatility_20d > volatility_high_threshold:
            reward -= 20  # Caution against buying in high volatility
        
    elif position == 1:  # Holding
        # Reward for holding during a strong trend
        if trend_r_squared > 0.8:
            reward += 30  # Encourage holding
        
        # Consider selling if trend weakens or overbought
        elif trend_r_squared < 0.5 or bb_pos > 0.8:  # Weak trend or overbought
            reward -= 30  # Consider selling
        
        # Penalty for holding in high volatility
        if volatility_5d > volatility_high_threshold:
            reward -= 20  # Caution in high volatility markets
            
    # Normalize reward to fit in the range [-100, 100]
    return np.clip(reward, -100, 100)