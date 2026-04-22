import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
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
        # Strong buy when in strong uptrend and oversold
        if trend_r_squared > 0.8 and bb_pos < 0.2:
            reward += 50  # Strong buy signal
        # Moderate buy when trend is moderate and oversold
        elif trend_r_squared > 0.5 and bb_pos < 0.3:
            reward += 30  # Potential buy signal
        # Caution in high volatility regime
        elif volatility_5d > high_vol_threshold:
            reward -= 20  # Penalty for buying in high volatility
        # Penalize buying at high levels without clear signals
        else:
            reward -= 10  # Neutral penalty for unclear conditions

    else:  # Holding stock (potential SELL)
        # Reward for holding in a strong trend
        if trend_r_squared > 0.8 and bb_pos < 0.75:
            reward += 20  # Positive reward for holding
        # Encourage selling in overbought conditions
        if bb_pos > 0.8:
            reward -= 40  # Strong penalty for holding in overbought
        # Encourage selling in weak trend
        elif trend_r_squared < 0.5:
            reward -= 30  # Penalty for holding in weak trend
        # Penalize holding during high volatility
        if volatility_20d > high_vol_threshold:
            reward -= 15  # Caution in high volatility

        # Additional encouragement to sell if trend is weakening
        if trend_r_squared < 0.6:
            reward -= 10  # Encourage evaluation for selling

    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward