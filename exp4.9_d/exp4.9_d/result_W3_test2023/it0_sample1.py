import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract features from the enhanced state
    s = enhanced_state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    
    # Historical volatility features
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    vol_ratio = s[138]  # Volume ratio (5d/20d)
    
    # Trend and regime features
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    
    # Reward initialization
    reward = 0
    
    # Define thresholds
    high_volatility_threshold = 2 * vol_20d  # High volatility is twice the 20-day volatility
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Overbought BB position threshold

    # Reward logic based on position
    if position == 0:  # Not holding
        # Identify clear BUY opportunities
        if trend_r2 > strong_trend_threshold:  # Clear upward trend
            reward += 50  # Strong trend gives a strong positive reward
        elif vol_ratio < 1:  # Low volume ratio indicates potential for upward movement
            reward += 20  # Potential bounce back considered as a positive signal
    else:  # Holding position
        # Encourage HOLD during uptrend
        if trend_r2 > strong_trend_threshold:
            reward += 30  # Positive reinforcement for holding in a strong trend
        # Encourage SELL when trend weakens or overbought conditions
        if trend_r2 < strong_trend_threshold or bb_pos > overbought_threshold:
            reward -= 40  # Negative reward for holding in a weak trend or overbought situation
    
    # Implement caution during extreme market conditions
    if vol_ratio > 2:  # Extreme market condition
        reward -= 30  # Caution signal triggers a negative adjustment

    # Normalize reward to the desired range [-100, 100]
    # Clamp the reward to ensure it stays within the range
    reward = np.clip(reward, -100, 100)
    
    return reward