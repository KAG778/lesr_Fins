import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Position flag
    holding = s[150]
    
    # Trend indicators
    trend_r2 = s[145]
    bb_pos = s[149]
    momentum = s[134]
    
    # Volatility indicators
    historical_volatility = s[135:137]
    vol_ratio = s[144]
    
    # Reward structure
    if holding == 0:  # Not holding (BUY phase)
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 20  # Caution to avoid buying
        
    else:  # Holding (SELL/HOLD phase)
        if trend_r2 > 0.8:  # Strong trend
            if bb_pos > 0.8:  # Overbought
                reward += 30  # Encouragement to sell
            else:  # Continue holding
                reward += 10  
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 25  # Encourage selling to avoid losses
        
    # Adjust for volatility regime
    if vol_ratio > 2.0:
        reward -= 10  # High caution in extreme volatility
    elif vol_ratio < 0.5:
        reward += 5  # Low volatility, potentially safer to trade
    
    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward