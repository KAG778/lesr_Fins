import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]
    
    # Price and trend indicators
    sma5 = s[120]
    sma20 = s[122]
    r_squared = s[145]
    bb_pos = s[149]
    
    # Volatility metrics
    historical_vol = np.mean(s[135:137])  # 5-day, 20-day historical volatility
    vol_ratio = s[144]  # Volatility regime ratio
    momentum = s[134]  # 10-day rate of change
    
    # Initialize reward
    reward = 0
    
    # Define thresholds based on historical volatility
    high_vol_threshold = 2 * historical_vol
    low_vol_threshold = 0.5 * historical_vol
    
    if position == 0:  # Not holding
        # Reward for BUY signals
        # Strong upward trend
        if sma5 > sma20 and r_squared > 0.8 and momentum > 0:
            reward += 50  # Strong buy opportunity
            
        # Oversold condition (RSI < 30)
        if s[128] < 30:
            reward += 30  # Buy opportunity due to oversold condition
            
        # Caution against extreme volatility
        if vol_ratio > 2:
            reward -= 20  # Exercise caution in high volatility market
            
    else:  # Holding
        # Reward for HOLD signals
        if sma5 > sma20 and r_squared > 0.8 and momentum > 0:
            reward += 20  # Continue holding during uptrend
            
        # Sell signals
        if bb_pos > 0.8:  # Overbought condition
            reward -= 50  # Consider selling
            
        # Weakening trend or volatility increase
        if r_squared < 0.5 or vol_ratio > 2:
            reward -= 30  # Consider selling due to trend weakness or high volatility
            
    # Normalize reward to ensure it remains within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward