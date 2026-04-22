import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0

    # Constants for calculations
    high_vol_threshold = 2.0  # High volatility ratio threshold
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    high_bb_threshold = 0.8  # High Bollinger Band position threshold
    
    # Calculate historical volatility
    vol_5d = s[135]
    vol_20d = s[136]
    average_vol = (vol_5d + vol_20d) / 2
    
    # Position flag
    holding_position = s[150]  # 1 if holding, 0 if not

    # Conditions for rewarding BUY signal
    if holding_position == 0:
        # Check if there's a strong uptrend (using SMA and RSI for confirmation)
        sma5 = s[120]
        sma20 = s[122]
        rsi5 = s[128]
        
        if sma5 > sma20 and rsi5 < 30:  # Oversold
            reward += 50  # Strong buy signal
        elif sma5 > sma20:  # General buy condition
            reward += 20  # Moderate buy signal
        
    # Conditions for rewarding HOLD signal
    elif holding_position == 1:
        trend_r2 = s[145]
        
        if trend_r2 > strong_trend_threshold:  # Strong trend
            reward += 30  # Reward holding in a strong uptrend
            
        # Check if conditions are ripe for a SELL signal
        bb_position = s[149]
        
        if bb_position > high_bb_threshold:  # Overbought condition
            reward -= 30  # Encourage selling to lock in profits
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 20  # Encourage selling to avoid losses
            
    # Adjust reward based on volatility regime
    vol_ratio = s[144]
    
    if vol_ratio > high_vol_threshold:  # Extreme market volatility
        reward -= 10  # Be cautious, reduce reward
    
    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward