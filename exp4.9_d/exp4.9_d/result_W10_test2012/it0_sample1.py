import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the state
    position = s[150]
    trend_r_squared = s[145]
    bb_position = s[149]
    volatility = s[135]
    atr = s[138]  # Average True Range can be used for volatility context
    momentum = s[134]
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds based on volatility
    high_volatility_threshold = 2 * np.mean([s[135], s[136]])  # Mean of 5-day and 20-day volatility
    overbought_threshold = 0.8
    strong_trend_threshold = 0.8
    
    if position == 0:  # Not holding
        # Positive reward for clear BUY opportunities
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 50  # Strong buy signal
        if bb_position < 0.2:  # Oversold condition
            reward += 30
        
        # Penalty for buying in high volatility
        if volatility > high_volatility_threshold:
            reward -= 20
            
    elif position == 1:  # Holding
        # Positive reward for HOLD during an uptrend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 20  # Encouraging to hold
        
        # Evaluate for SELL opportunities
        if trend_r_squared < strong_trend_threshold or bb_position > overbought_threshold:
            reward += 30  # Clear signal to sell
        if bb_position > overbought_threshold and momentum < 0:
            reward += 50  # Strong sell signal
        
        # Penalty for selling in low volatility
        if volatility < high_volatility_threshold:
            reward -= 20
    
    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward