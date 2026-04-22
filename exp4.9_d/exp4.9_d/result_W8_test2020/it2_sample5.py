import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # High volatility condition
    low_volatility_threshold = 0.5 * avg_volatility  # Low volatility condition

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding stock (BUY conditions)
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Decent trend and slightly oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # High volatility caution
            reward -= 20  # Penalize for buying in high volatility
        elif volatility_5d < low_volatility_threshold:  # Low volatility condition
            reward += 10  # Encourage buying in low volatility
        else:
            reward -= 5  # Neutral case (slight penalty for indecision)

    elif position == 1:  # Holding stock (SELL conditions)
        if trend_r2 > 0.8:  # Strong trend, encourage holding
            reward += 20  # Reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Strong signal to consider selling
        elif trend_r2 < 0.5:  # Weak trend, consider selling
            reward -= 40  # Strong signal to sell
        elif volatility_5d > high_volatility_threshold:  # Caution in high volatility
            reward -= 20  # Penalize holding in high volatility
        else:
            reward += 5  # Mild reward for stable holding conditions

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward