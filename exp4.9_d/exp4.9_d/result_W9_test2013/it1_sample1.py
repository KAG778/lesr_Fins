import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    low_vol_threshold = 1.0 * volatility_5d  # 1x 5-day volatility
    high_vol_threshold = 2.0 * volatility_20d  # 2x 20-day volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold

    if position == 0:  # Not holding
        # Reward for clear BUY opportunities
        if trend_r2 > strong_trend_threshold and bb_pos < oversold_threshold:
            reward += 50  # Strong buy signal in a strong uptrend
        elif trend_r2 > 0.6 and bb_pos < 0.4:
            reward += 30  # Moderate buy signal
        elif bb_pos > overbought_threshold:  
            reward -= 20  # Penalize buying in overbought conditions
        if volatility_5d > high_vol_threshold:
            reward -= 15  # Caution against buying in high volatility

    else:  # Holding
        # Reward for HOLD during strong trend
        if trend_r2 > strong_trend_threshold:
            reward += 20  # Reward for holding in a strong uptrend
        elif trend_r2 < 0.5 or bb_pos > overbought_threshold:
            reward -= 50  # Strong sell signal when trend weakens or overbought
        elif bb_pos > 0.6 and trend_r2 < 0.7:
            reward -= 20  # Consider selling when moderately overbought with weak trend
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Caution in high volatility while holding

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward