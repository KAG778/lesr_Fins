import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
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
        # Strong buy signal conditions
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_pos < 0.3:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy signal
        elif bb_pos < 0.15 and volatility_5d < low_vol_threshold:  # Strongly oversold with low volatility
            reward += 40  # Additional buy signal
        elif volatility_5d > high_vol_threshold:  # High volatility caution
            reward -= 15  # Penalty for buying in high volatility

    else:  # Holding stock (potential SELL)
        # Reward for holding during a strong trend
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Reward for holding in strong trend
        # Conditions for selling
        if bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong penalty for holding in overbought
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Encourage selling
        elif trend_r_squared < 0.6 and bb_pos > 0.6:  # Weak trend and near overbought
            reward -= 20  # Additional penalty for selling in uncertain conditions

        # Caution for high volatility
        if volatility_5d > high_vol_threshold:  # High volatility environment
            reward -= 15  # Caution in extreme volatility

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward