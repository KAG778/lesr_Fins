import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    high_vol_threshold = 2 * volatility_20d  # High volatility condition
    low_vol_threshold = 0.5 * volatility_20d  # Low volatility condition

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock (BUY opportunity)
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Slightly weaker conditions
            reward += 30  # Mild buy signal
        elif volatility_5d > high_vol_threshold:  # High volatility, be cautious
            reward -= 20  # Negative reward for buy in high volatility
        else:
            reward -= 5  # Neutral case, slight penalty for indecision

    elif position == 1:  # Holding stock (SELL opportunity)
        if trend_r2 > 0.8 and bb_pos < 0.8:  # Strong trend but not overbought
            reward += 20  # Encourage holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Strong signal to consider selling
        elif trend_r2 < 0.5:  # Weak trend, consider selling
            reward -= 40  # Strong signal to sell
        elif volatility_5d > high_vol_threshold:  # Caution in high volatility
            reward -= 20  # Penalize holding in high volatility
        else:
            reward += 10  # Neutral holding position

    # Normalize the reward to the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward