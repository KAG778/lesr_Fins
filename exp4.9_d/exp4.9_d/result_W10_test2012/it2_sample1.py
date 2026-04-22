import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the state
    position = enhanced_state[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_strength = enhanced_state[145]  # Trend strength (R² of regression)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Define relative volatility thresholds
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    strong_trend_threshold = 0.8
    moderate_trend_threshold = 0.5
    overbought_threshold = 0.8
    oversold_threshold = 0.2
    
    # Initialize the reward
    reward = 0

    if position == 0:  # Not holding (potential buy signals)
        # Encourage buying on strong uptrend and oversold condition
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > moderate_trend_threshold and bb_position < 0.4:
            reward += 30  # Moderate buy signal

        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in volatile conditions

        # Encourage buying in low volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Encouragement in low volatility

    elif position == 1:  # Holding (potential sell signals)
        # Encourage holding in a strong uptrend
        if trend_strength > strong_trend_threshold:
            reward += 20  # Encourage holding

        # Evaluate for sell signals
        if bb_position > overbought_threshold or trend_strength < moderate_trend_threshold:
            reward -= 40  # Strong sell signal

        # Penalize holding in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 30  # Caution advised for holding in high volatility

        # Mild reward for holding in stable conditions
        elif volatility_5d < low_volatility_threshold:
            reward += 5  # Mild reward for holding in stable conditions

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward