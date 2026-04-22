import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features
    position = enhanced_state[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_pos = enhanced_state[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    low_vol_threshold = 0.5 * avg_volatility
    high_vol_threshold = 2.0 * avg_volatility

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock (potential BUY)
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_pos < 0.25:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        # Moderate buy signal
        elif trend_r_squared > 0.6 and bb_pos < 0.4:
            reward += 30  # Moderate buy signal
        # Caution in high volatility
        elif volatility_5d > high_vol_threshold:
            reward -= 20  # Penalty for buying in high volatility
        # Neutral situation
        else:
            reward -= 5  # Slight penalty for unclear conditions

    else:  # Holding stock (potential SELL)
        # Reward for holding in strong trend
        if trend_r_squared > 0.8 and bb_pos < 0.7:
            reward += 30  # Positive reward for holding
        # Encourage selling in overbought conditions
        elif bb_pos > 0.8:
            reward -= 40  # Strong penalty for holding in overbought
        # Encourage selling in weak trend
        elif trend_r_squared < 0.5:
            reward -= 30  # Penalty for holding in weak trend
        # Caution in high volatility
        if volatility_20d > high_vol_threshold:
            reward -= 15  # Caution in high volatility
    
    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward