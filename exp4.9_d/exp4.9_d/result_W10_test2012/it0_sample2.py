import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract features
    close_prices = s[0:20]
    position = s[150]
    
    # Calculate historical volatility from the provided features
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Define thresholds based on volatility
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Regime features
    trend_r_squared = s[145]   # Trend strength (R² of regression)
    bb_position = s[149]       # Bollinger Band position [0,1]
    
    reward = 0
    
    if position == 0:  # Not holding
        # Conditions for buying
        if trend_r_squared > 0.8:  # Clear trend
            if bb_position < 0.2:  # Oversold condition
                reward += 50  # Strong buy signal
            elif bb_position < 0.4:  # Reasonable buy signal
                reward += 20
        
        # Penalize for buying in high volatility regime
        if avg_volatility > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    else:  # Holding
        # Conditions for holding
        if trend_r_squared > 0.8:  # Clear trend
            reward += 30  # Reward for holding in an uptrend
        
        # Conditions for selling
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 50  # Strong sell signal
        elif bb_position > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
        
        # Penalize for holding in high volatility regime
        if avg_volatility > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    # Ensure the reward stays within bounds
    reward = np.clip(reward, -100, 100)
    
    return reward