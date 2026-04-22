import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define reward variables
    reward = 0

    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility regime
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold

    # Reward logic based on the position
    if position == 0:  # Not holding
        # Strong buy signal
        if bb_position < oversold_threshold and trend_r_squared > strong_trend_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif bb_position < 0.5 and trend_r_squared > strong_trend_threshold:
            reward += 20  # Moderate buy opportunity
        # Caution in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Discourage buying in high volatility
    else:  # Holding
        # Reward for holding during a strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding
        # Encourage selling when overbought or trend weakens
        if bb_position > overbought_threshold or trend_r_squared < 0.5:
            reward -= 30  # Strong incentive to consider selling
        # Penalize if overbought but still holding
        if bb_position > overbought_threshold:
            reward -= 10  # Additional caution penalty

    # Normalize to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward