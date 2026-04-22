import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state

    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2

    # Define thresholds
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    strong_trend_threshold = 0.8
    moderate_trend_threshold = 0.6
    overbought_threshold = 0.8
    oversold_threshold = 0.2

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Strong buy signal conditions
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > moderate_trend_threshold and bb_position < 0.4:
            reward += 30  # Moderate buy signal

        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 30  # Caution against buying in volatile conditions
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encouragement in low volatility

    elif position == 1:  # Holding
        # Reward for holding in strong uptrend
        if trend_strength > strong_trend_threshold:
            reward += 25  # Encourage holding
        elif trend_strength < moderate_trend_threshold or bb_position > overbought_threshold:
            reward -= 50  # Strong signal to sell
        elif bb_position > 0.8:  
            reward -= 30  # Clear signal to consider selling

        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution advised for holding in high volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 5  # Mild reward for holding in stable conditions

    # Ensure reward is within bounds [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward