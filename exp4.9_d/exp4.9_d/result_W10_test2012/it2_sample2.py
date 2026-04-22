import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state

    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2

    # Define thresholds
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    overbought_threshold = 0.8
    oversold_threshold = 0.2
    strong_trend_threshold = 0.8

    reward = 0.0

    if position == 0:  # Not holding
        # Conditions for buying
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > 0.6 and bb_position < 0.4:
            reward += 30  # Moderate buy opportunity

        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution advised
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encouraging buying in low volatility

    elif position == 1:  # Holding
        # Conditions for holding
        if trend_strength > strong_trend_threshold:
            reward += 20  # Encourage holding in a strong trend
        elif trend_strength < 0.5 or bb_position > overbought_threshold:
            reward -= 40  # Strong signal to sell
        elif bb_position > overbought_threshold and trend_strength < strong_trend_threshold:
            reward -= 30  # Consider selling in overbought conditions

        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against holding in volatile conditions

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward