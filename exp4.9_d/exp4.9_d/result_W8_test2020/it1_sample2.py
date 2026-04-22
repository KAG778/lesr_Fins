import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Extreme market condition
    low_volatility_threshold = 0.5 * avg_volatility  # Safe entry condition

    # Initialize reward
    reward = 0.0

    # Reward logic based on position and market conditions
    if position == 0:  # Not holding
        # Check for buy conditions
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Good trend but not oversold
            reward += 30  # Good buy conditions
        elif trend_r2 < 0.5 and volatility_5d > high_volatility_threshold:  # Weak trend and high volatility
            reward -= 20  # Negative reward, be cautious about buying
        elif volatility_5d < low_volatility_threshold:  # Low volatility
            reward += 20  # Safe to enter a position
        else:
            reward -= 5  # Slight penalty for indecision

    elif position == 1:  # Holding
        # Check for hold or sell conditions
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 40  # Strong signal to sell
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Be cautious, consider selling
        else:
            reward += 5  # Mild reward for holding in stable conditions

    # Normalize reward to the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward