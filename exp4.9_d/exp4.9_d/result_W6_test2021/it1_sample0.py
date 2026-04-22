import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_pos = enhanced_state[149]  # Bollinger Band position [0,1]
    regime_volatility = enhanced_state[144]  # Volatility regime ratio

    # Calculate thresholds based on volatility
    low_vol_threshold = 0.5 * volatility_20d
    high_vol_threshold = 2.0 * volatility_20d

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock (potential BUY)
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_pos < 0.25:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_pos < 0.3:  # Moderate buy signal
            reward += 30  # Moderate buy signal
        elif bb_pos < 0.2 and volatility_5d < low_vol_threshold:  # Oversold with low volatility
            reward += 40  # Strong oversold condition
        elif bb_pos > 0.8 and volatility_5d > high_vol_threshold:  # Overbought in high volatility
            reward -= 20  # Caution in overbought condition

    else:  # Holding stock (potential SELL)
        # Reward for holding during a strong trend
        if trend_r_squared > 0.8 and bb_pos < 0.75:  # Strong trend, not overbought
            reward += 20  # Positive reward for holding
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Penalty for holding in weak trend
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Penalty for holding in overbought condition
        elif volatility_20d > high_vol_threshold:  # High volatility environment
            reward -= 15  # Caution in high volatility

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward