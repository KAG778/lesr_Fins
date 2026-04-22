import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is a 151-dimensional array
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define reward variable
    reward = 0
    
    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility regime
    low_volatility_threshold = 0.5 * volatility_20d   # Low volatility regime
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition
    
    if position == 0:  # Not holding (potential buy)
        # Conditions for buying
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4: 
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:
            reward -= 15  # Discourage buying in high volatility
        elif trend_r_squared < 0.4: 
            reward -= 10  # Penalty for indecisive market
    
    else:  # Holding (position = 1)
        # Reward for holding during a strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Reward for holding
        elif trend_r_squared < 0.5 or bb_position > overbought_threshold:
            reward -= 40  # Strong signal to consider selling
        if bb_position > overbought_threshold:
            reward -= 10  # Additional caution penalty
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Caution in high volatility

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward