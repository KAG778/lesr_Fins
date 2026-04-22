import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract features from the enhanced_state
    s = enhanced_state
    position = s[150]  # Current position (1.0 = holding stock, 0.0 = not holding)
    
    # Volatility features
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136] # 20-day historical volatility
    regime_volatility = s[144]  # Volatility regime ratio
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    reward = 0.0  # Initialize reward
    
    if position == 0.0:  # Not holding stock
        # Reward for buying opportunities
        if trend_strength > 0.8 and bb_position < 0.2:  # Clear uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_strength <= 0.8 and bb_position < 0.5:  # Moderate trend
            reward += 20  # Moderate buy signal
        
    elif position == 1.0:  # Holding stock
        # Reward for holding during an uptrend
        if trend_strength > 0.8:  # Strong trend
            reward += 30  # Hold in strong trend
        elif trend_strength < 0.5 and bb_position > 0.8:  # Uptrend weakening and overbought
            reward += -40  # Consider selling
        
        # Encourage selling when trend weakens
        if trend_strength < 0.5:  # Weak trend
            reward += -20  # Sell signal due to weakness
            
    # Caution in extreme volatility regimes
    if regime_volatility > 2.0:
        reward += -10  # Caution penalty in extreme volatility
    
    # Scale the reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward