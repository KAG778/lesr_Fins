import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    price_pos = s[146]  # Price position in the 20-day range [0, 1]
    
    # Initialize reward
    reward = 0

    # Define thresholds based on historical volatility
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility threshold
    moderate_volatility_threshold = 1.0 * volatility_20d  # Moderate volatility threshold
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold

    if position == 0:  # Not holding (potential buy)
        # Strong buy opportunity
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        # Moderate buy opportunity
        elif trend_r_squared > 0.6 and bb_position < 0.5:
            reward += 30  # Moderate buy signal
        # Penalty for buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against buying in high volatility
        # Caution for weak trends
        elif trend_r_squared < 0.4:
            reward -= 10  # Penalty for indecisive market

    else:  # Holding (position = 1.0)
        # Reward for holding during strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding 
        # Encourage selling when overbought
        if bb_position > overbought_threshold:
            reward -= 30  # Strong incentive to consider selling
        # Encourage selling if trend weakens
        elif trend_r_squared < 0.5:
            reward -= 50  # Strong sell signal
        # Moderate penalty for high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Caution against high volatility

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward