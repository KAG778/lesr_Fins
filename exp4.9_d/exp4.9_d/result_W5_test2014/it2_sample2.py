import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define thresholds
    high_volatility_threshold = 2 * volatility_20d  # High volatility caution threshold
    low_volatility_threshold = 0.5 * volatility_20d  # Low volatility caution threshold
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB
    overbought_threshold = 0.8  # Overbought condition for BB

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding (potential buy)
        # Strong buy conditions
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.5 and bb_position < 0.3:
            reward += 30  # Moderate buy signal

        # Adjust for volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    elif position == 1:  # Holding (potential sell)
        # Reward for holding in a strong trend
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Positive reinforcement for holding
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Encourage selling
        
        # Sell conditions
        if bb_position > overbought_threshold:
            reward -= 40  # Strong signal to sell

        # Adjust for volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution for holding in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward