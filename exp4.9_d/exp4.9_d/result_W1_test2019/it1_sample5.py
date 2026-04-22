import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_position = s[149]  # Bollinger Band position
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Threshold for caution
    low_volatility_threshold = 0.5 * avg_volatility  # Threshold for cautious buying
    
    # Initialize reward
    reward = 0.0
    
    if position == 0:  # Not holding - BUY opportunities
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_position < 0.4:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy opportunity
        elif bb_position < 0.3 and volatility_5d < low_volatility_threshold:  # Oversold with low volatility
            reward += 20  # Buy with low volatility
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 10  # Caution on buying
    
    elif position == 1:  # Holding - SELL or HOLD opportunities
        if trend_r_squared > 0.8:  # Strong trend
            reward += 10  # Reward for holding in strong trend
        elif trend_r_squared < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 30  # Strong signal to consider selling
        elif bb_position > 0.7 and volatility_5d > high_volatility_threshold:  # Overbought in high volatility
            reward -= 50  # Strong sell signal
        elif bb_position > 0.6:  # Caution in slightly overbought conditions
            reward -= 20  # Caution against holding
    
    # Limit extreme trading actions based on volatility regime
    if volatility_5d > high_volatility_threshold or volatility_20d > high_volatility_threshold:
        reward -= 20  # Caution in extreme volatility conditions
    
    # Normalize the reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward