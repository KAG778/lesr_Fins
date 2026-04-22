import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    
    # Extract relevant features
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Extreme market condition
    low_volatility_threshold = 0.5 * avg_volatility  # Low volatility condition

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding stock
        # Look for BUY signals
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Good trend and near oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # Caution in high volatility
            reward -= 20  # Penalize entering positions under high volatility
        else:
            reward -= 5  # Neutral case (slight penalty for indecision)

    else:  # Holding stock
        # Look for HOLD or SELL signals
        if trend_r2 > 0.8:  # Very strong trend
            reward += 20  # Reward for holding
        elif bb_pos > 0.8:  # Overbought conditions
            reward -= 30  # Strong sell signal
        elif trend_r2 < 0.5 or bb_pos > 0.7:  # Weak trend or high bb position
            reward -= 40  # Strongly consider selling
        elif volatility_5d > high_volatility_threshold:  # Caution when holding in high volatility
            reward -= 10  # Penalize holding in high volatility
        else:
            reward += 5  # Neutral reward for holding in stable conditions

    # Ensure that the reward is within the specified range
    reward = np.clip(reward, -100, 100)

    return reward