import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # Current position (holding or not)
    
    # Trend and Regime Features
    trend_strength = s[145]  # Trend R²
    bb_position = s[149]  # Bollinger Band position
    historical_volatility = np.mean(s[135:137])  # Average of 5-day and 20-day historical volatility
    
    reward = 0
    
    if position == 1:  # Holding stock
        # Reward for holding during strong uptrend
        if trend_strength > 0.8:
            reward += 10  # Strong trend holding reward
        # Penalize for holding during weak trend
        elif trend_strength < 0.5:
            reward -= 20  # Weak trend penalty
        
        # Consider selling if overbought
        if bb_position > 0.8:
            reward += 20  # Encourage sell in overbought conditions
            
    else:  # Not holding stock
        # Reward for buying in oversold condition
        if trend_strength > 0.8 and bb_position < 0.2:
            reward += 20  # Clear buy opportunity
        
        # Penalize for buying in overbought condition
        if bb_position > 0.8:
            reward -= 10  # Discourage buying in overbought
        
    # Normalize to be in [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward