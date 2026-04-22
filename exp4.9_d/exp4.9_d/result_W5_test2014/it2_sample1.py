import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Position flag (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_position = s[149]  # Bollinger Band position [0, 1]

    # Define thresholds based on volatility and trends
    high_volatility_threshold = 1.5 * volatility_20d  # Caution threshold for high volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for Bollinger Bands
    overbought_threshold = 0.8  # Overbought condition for Bollinger Bands

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding
        # Strong buy signal conditions
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > 0.6 and bb_position < 0.3:  # Moderately strong trend
            reward += 30  # Moderate buy opportunity

        # Penalize potential buying in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution on buying in high volatility

    elif position == 1:  # Holding
        # Reward for holding during strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Positive reinforcement for holding
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 50  # Encourage selling in weak trend
        
        # Manage overbought conditions
        if bb_position > overbought_threshold:  # Overbought condition
            reward -= 40  # Strong signal to sell
        
        # Penalize holding during high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution on holding in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward