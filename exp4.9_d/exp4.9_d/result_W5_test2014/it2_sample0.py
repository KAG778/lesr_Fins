import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149]  # Bollinger Band position [0,1]
    
    # Define thresholds based on historical volatility
    high_volatility_threshold = 1.5 * volatility_20d  # High volatility threshold for caution
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB
    overbought_threshold = 0.8  # Overbought condition for BB

    reward = 0.0

    if position == 0:  # Not holding the stock
        # Strong buy signal conditions
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > 0.6 and bb_position < 0.3:
            reward += 30  # Moderate buy opportunity
        
        # Penalize potential buying in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution on buying in high volatility

    elif position == 1:  # Holding the stock
        # Reward for holding during strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Encourage holding in a strong trend
        elif trend_r_squared < 0.5:  # Weakening trend
            reward -= 40  # Strong signal to consider selling
        
        # Overbought condition
        if bb_position > overbought_threshold:  # Strong incentive to sell
            reward -= 30  
        
        # Penalize holding during high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution on holding during high volatility

    # Normalize the reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward