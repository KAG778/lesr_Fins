import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]
    volatility_5d = s[135]
    volatility_20d = s[136]
    r_squared = s[145]
    bb_position = s[149]
    rsi_5d = s[128]  # Assuming this is the 5-day RSI
    momentum = s[134]
    
    # Initialize reward
    reward = 0
    
    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # Adjust based on historical volatility
    strong_trend_threshold = 0.8  # R² threshold for strong trends
    overbought_threshold = 0.8  # BB position for overbought conditions
    
    # Reward logic based on position
    if position == 0:  # Not holding stock
        # Positive reward for buying opportunities
        if rsi_5d < 30 and momentum > 0:  # Oversold condition and positive momentum
            reward += 50  # Strong buy signal
        if r_squared > strong_trend_threshold:  # Strong uptrend
            reward += 30  # Considered a buy signal
    else:  # Holding stock
        # Reward for holding during strong trends
        if r_squared > strong_trend_threshold:
            reward += 20  # Strong uptrend, hold
        
        # Penalize for selling during strong trends
        if bb_position > overbought_threshold:
            reward -= 30  # Considered a sell signal, risk of reversal
        
        # Encourage selling when conditions weaken
        if r_squared < 0.5:  # Weakening trend
            reward += 10  # Consider selling, but not too aggressive
        
    # Penalize for extreme volatility
    if volatility_5d > high_volatility_threshold:
        reward -= 20  # Caution in extreme volatility
    
    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward