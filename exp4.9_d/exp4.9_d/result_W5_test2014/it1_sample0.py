import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define thresholds based on historical volatility
    high_volatility_threshold = 2 * volatility_20d  # For buying
    low_volatility_threshold = 1.5 * volatility_5d  # For selling
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB
    overbought_threshold = 0.8  # Overbought condition for BB

    reward = 0

    if position == 0:  # Not holding
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif bb_position < oversold_threshold:
            reward += 30  # Moderate buy signal
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    elif position == 1:  # Holding
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Encourage holding in strong trend
        if bb_position > overbought_threshold:
            reward -= 40  # Strong sell signal
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Encourage selling
        if volatility_20d > low_volatility_threshold:
            reward -= 15  # Caution in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward