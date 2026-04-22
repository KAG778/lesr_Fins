import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # R² of the trend
    bb_position = s[149]  # Bollinger Band position [0,1]
    vol_ratio = s[144]  # Volatility regime ratio

    # Define thresholds
    high_volatility_threshold = 2 * np.mean([volatility_5d, volatility_20d])  # 2x average volatility
    low_volatility_threshold = 0.5 * np.mean([volatility_5d, volatility_20d])  # 0.5x average volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition

    if position == 0:  # Not holding (BUY opportunities)
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r_squared > strong_trend_threshold and bb_position < 0.5:
            reward += 30  # Moderate buy opportunity in a strong trend
        elif vol_ratio < 0.5:  # Low volatility
            reward += 20  # Reward for trading in calm market conditions
        elif bb_position < oversold_threshold:
            reward += 10  # Buying opportunity based on oversold condition
        
    elif position == 1:  # Holding (SELL opportunities)
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Encourage holding in a strong trend
        elif trend_r_squared < 0.5:  # Weak trend
            if bb_position > overbought_threshold:
                reward -= 40  # Strong signal to sell in overbought
            elif vol_ratio > high_volatility_threshold:
                reward -= 20  # Caution against holding in high volatility
            else:
                reward -= 10  # Neutral signal for weak trend
        elif bb_position > overbought_threshold:
            reward -= 30  # Consider selling in overbought condition

    # Adjust for extreme volatility
    if vol_ratio > high_volatility_threshold:
        reward -= 20  # Penalty for trading in high volatility

    # Normalize the reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward