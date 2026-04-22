import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r2 = enhanced_state[145]  # Trend strength (R² of regression)
    bb_pos = enhanced_state[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # High volatility condition
    low_volatility_threshold = 0.5 * avg_volatility  # Low volatility condition

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding stock (BUY conditions)
        # Look for strong buy signals
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Good trend but not oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # High volatility caution
            reward -= 30  # Penalize for buying in high volatility
        else:
            reward -= 5  # Neutral case, slight penalty for indecision

    elif position == 1:  # Holding stock (SELL conditions)
        # Look for sell or hold signals
        if trend_r2 < 0.5 or bb_pos > 0.8:  # Weak trend or overbought
            reward -= 50  # Strong sell signal
        elif trend_r2 > 0.8:  # Strong trend, reward for holding
            reward += 20  # Reward for holding in a strong trend
        elif bb_pos > 0.7:  # Nearing overbought
            reward -= 20  # Consider selling
        elif volatility_5d > high_volatility_threshold:  # High volatility caution
            reward -= 20  # Penalize holding in high volatility
        else:
            reward += 5  # Mild reward for stable holding conditions

    # Normalize reward to be within [-100, 100]
    return np.clip(reward, -100, 100)