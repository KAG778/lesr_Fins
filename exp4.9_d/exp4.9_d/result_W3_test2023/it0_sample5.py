import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features from the enhanced state
    position = s[150]
    
    # Volatility measures
    historical_volatility_5d = s[135]
    historical_volatility_20d = s[136]
    avg_historical_volatility = (historical_volatility_5d + historical_volatility_20d) / 2
    
    # Trend measures
    trend_r_squared = s[145]
    momentum = s[134]
    
    # Bollinger Band position
    bb_position = s[149]
    
    # RSI metrics
    rsi_5d = s[128]
    rsi_10d = s[129]
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on position
    if position == 0:  # Not holding
        # Buy signal conditions
        if trend_r_squared > 0.8 and rsi_5d < 30 and bb_position < 0.2:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > 0.6 and rsi_5d < 40:
            reward += 20  # Moderate buy opportunity
        elif avg_historical_volatility > 2 * historical_volatility_5d:
            reward -= 20  # High volatility, cautious
    else:  # Holding
        # Hold signal conditions
        if trend_r_squared > 0.8 and momentum > 0:
            reward += 30  # Good to hold
        elif trend_r_squared < 0.5 or rsi_5d > 70 or bb_position > 0.8:
            reward += 20  # Time to consider selling
        elif trend_r_squared < 0.4:
            reward -= 30  # Weak trend, consider selling
    
    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward