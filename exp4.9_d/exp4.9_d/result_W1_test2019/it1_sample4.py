import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]
    volatility_20d = s[136]
    trend_r_squared = s[145]
    bb_position = s[149]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = avg_volatility / 2
    
    # Define thresholds
    strong_trend_threshold = 0.8  # Strong trend R²
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition

    if position == 0:  # Not holding (Buy opportunities)
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r_squared > strong_trend_threshold and bb_position < 0.5:
            reward += 30  # Moderate buy signal
        elif bb_position < oversold_threshold:
            reward += 20  # Buy opportunity based on oversold condition
        else:
            reward -= 10  # Neutral penalty for unclear signals

    elif position == 1:  # Holding (Sell or Hold opportunities)
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Encourage holding in a strong trend
        elif trend_r_squared < 0.5 and bb_position > overbought_threshold:
            reward -= 50  # Strong sell opportunity
        elif trend_r_squared < 0.5:
            reward -= 20  # Weak trend, consider selling
        elif bb_position > overbought_threshold:
            reward -= 30  # Overbought condition, consider selling
        else:
            reward += 10  # Neutral holding

    # Adjust for market volatility
    if volatility_20d > high_volatility_threshold:
        reward -= 20  # Caution in extreme volatility
    elif volatility_20d < low_volatility_threshold:
        reward += 10  # Reward for trading in low volatility conditions

    # Normalize the reward to fit in the range of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward