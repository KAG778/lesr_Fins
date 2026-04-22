import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]
    
    reward = 0.0
    
    # Extract relevant features
    rsi_5 = s[128]
    momentum = s[134]
    trend_r_squared = s[145]
    bb_pos = s[149]
    volatility_5d = s[135]
    volatility_20d = s[136]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d != 0 else 0

    if position == 0:  # Not holding
        # Reward for clear BUY opportunities
        if rsi_5 < 30 and momentum > 0 and trend_r_squared > 0.8:
            reward += 50  # Strong buy signal
        elif volatility_ratio < 2 and bb_pos < 0.2:
            reward += 20  # Additional buying opportunity
        
        # Penalty for buying in high volatility
        if volatility_ratio > 2:
            reward -= 10  # Caution in extreme volatility

    elif position == 1:  # Holding
        # Reward for holding in an uptrend
        if trend_r_squared > 0.8 and momentum > 0:
            reward += 30  # Strong trend, continue holding
        elif bb_pos > 0.8 or rsi_5 > 70:
            reward -= 20  # Consider selling in overbought conditions
        
        # Encouragement to sell if trend weakens
        if momentum < 0 or trend_r_squared < 0.5:
            reward -= 30  # Weak trend, consider selling

    # Scale reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward