import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Extreme market condition

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock
        # Look for BUY signals
        if trend_r2 > 0.8:  # Strong trend
            reward += 50  # Positive reward for following trend
        elif bb_pos < 0.2:  # Oversold condition
            reward += 40  # Strong buy opportunity
        elif volatility_5d < avg_volatility:  # Low volatility
            reward += 30  # Safe to enter a position
        else:
            reward -= 10  # Neutral or weak conditions

    elif position == 1:  # Holding stock
        # Look for HOLD or SELL signals
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Positive reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Be cautious, consider selling
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 40  # Weak trend, consider selling
        else:
            reward += 10  # Neutral holding position

    # Scale the reward to fit within the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward