import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant state features
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Define thresholds based on volatility
    low_vol_threshold = 1.0 * volatility_20d
    high_vol_threshold = 2.0 * volatility_20d
    
    reward = 0
    
    if position == 0:  # Not holding stock
        # Conditions for potential BUY
        if trend_r_squared > 0.8:  # Strong trend
            reward += 50  # Positive reward for strong uptrend
        if bb_position < 0.2:  # Oversold condition
            reward += 30  # Positive reward for oversold
        if volatility_5d < low_vol_threshold:  # Low volatility
            reward += 20  # Positive reward for stability
    else:  # Holding stock
        # Conditions for potential HOLD
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Positive reward for holding in strong trend
        # Conditions for potential SELL
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 50  # Penalty for holding in weak trend
        if bb_position > 0.8:  # Overbought condition
            reward -= 30  # Penalty for overbought
        if volatility_5d > high_vol_threshold:  # High volatility
            reward -= 20  # Caution during high volatility
            
    # Ensure the reward is within the range [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward