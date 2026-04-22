import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0

    position = s[150]
    
    trend_r_squared = s[145]
    bollinger_band_position = s[149]
    volatility_ratio = s[135] / s[137]  # Ratio of 5-day to 20-day volatility
    rsi_14 = s[131]

    # Define thresholds
    high_trend_threshold = 0.8
    low_bb_threshold = 0.2
    high_bb_threshold = 0.8
    high_volatility_threshold = 2.0

    if position == 0:  # Not holding
        if trend_r_squared > high_trend_threshold and bollinger_band_position < low_bb_threshold:
            reward += 50  # Strong buy signal
        elif bollinger_band_position < low_bb_threshold and rsi_14 < 30:
            reward += 30  # Oversold condition buy signal
        else:
            reward -= 10  # Neutral to discourage random buys
    else:  # Holding
        if trend_r_squared > high_trend_threshold:
            reward += 10  # Encourage holding in a strong trend
        if bollinger_band_position > high_bb_threshold:
            reward += 20  # Consider selling when overbought
        if volatility_ratio > high_volatility_threshold:
            reward -= 20  # Penalize holding in high volatility
        if trend_r_squared < 0.5:
            reward -= 30  # Weak trend, consider selling

    # Clip the reward to the range of [-100, 100]
    return np.clip(reward, -100, 100)