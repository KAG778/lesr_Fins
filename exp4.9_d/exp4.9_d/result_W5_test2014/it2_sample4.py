import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define relative thresholds
    high_volatility_threshold = 2 * volatility_20d  # Caution threshold for buying/selling
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB
    overbought_threshold = 0.8  # Overbought condition for BB

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding the stock
        # Strong buying opportunities
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.3:
            reward += 30  # Moderate buy signal
        
        # Penalize for buying in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution on buying
        
    elif position == 1:  # Holding the stock
        # Reward for holding in strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Positive reinforcement for holding
        elif trend_r_squared < 0.5 or bb_position > overbought_threshold:
            reward -= 50  # Strong sell signal
        
        # Penalize holding during high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution on holding

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward