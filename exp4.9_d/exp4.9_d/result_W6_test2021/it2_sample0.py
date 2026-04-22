import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]

    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_vol_threshold = 2.0 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock (potential BUY)
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.3:  # Moderate trend and somewhat oversold
            reward += 30  # Moderate buy signal
        elif bb_position < 0.2 and volatility_5d < low_vol_threshold:  # Strong oversold with low volatility
            reward += 40  # Strong oversold condition
        elif bb_position > 0.8 and volatility_5d > high_vol_threshold:  # Overbought with high volatility
            reward -= 20  # Caution in overbought condition
        else:
            reward -= 5  # Minor penalty for unclear conditions

    else:  # Holding stock (potential SELL)
        # Reward for holding in a strong trend
        if trend_r_squared > 0.8 and bb_position < 0.75:  # Strong trend and not overbought
            reward += 20  # Positive reward for holding
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Strong penalty for holding in weak trend
        elif bb_position > 0.8:  # Overbought condition
            reward -= 40  # Strong penalty for holding in overbought
        elif volatility_20d > high_vol_threshold:  # High volatility environment
            reward -= 15  # Caution in high volatility
        else:
            reward += 10  # Small positive for holding in uncertain conditions

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward