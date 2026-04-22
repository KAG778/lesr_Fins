import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio
    
    # Initialize the reward
    reward = 0.0
    
    # Calculate dynamic thresholds
    avg_volatility = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    if position == 0:  # Not holding (BUY phase)
        # Conditions for a strong buy signal
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
            
        elif trend_r2 > 0.7 and bb_pos < 0.3:  # Moderate trend, slightly oversold
            reward += 30  # Moderate buy signal
            
        # Caution in high volatility environments
        if vol_ratio > 2.0:
            reward -= 20  # Caution in high volatility
        
    else:  # Holding (SELL phase)
        # Conditions for a strong hold signal
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Encourage holding
            
        # Conditions for a sell signal
        if bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
            
        if trend_r2 < 0.5:  # Weak trend
            reward -= 40  # Strong sell signal
            
        # Caution in high volatility
        if vol_ratio > 2.0:
            reward -= 30  # Caution in high volatility
        
        # Encourage selling if price is declining significantly
        if s[0] < s[1]:  # Current price lower than previous price
            reward -= 10  # Small penalty for price drop
            
    # Normalize the reward to fit the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward