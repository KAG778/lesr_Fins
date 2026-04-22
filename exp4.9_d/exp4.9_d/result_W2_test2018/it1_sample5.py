import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # Position flag (1.0 = holding, 0.0 = not holding)
    
    # Volatility features
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_volatility = (vol_5d + vol_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Trend and Bollinger Band features
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_position = s[149]  # Bollinger Band position
    price_position = s[146]  # Price position in 20-day range
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        # Moderate buy signal
        elif trend_r_squared > 0.6 and price_position < 0.4:  # Moderate trend and near oversold
            reward += 30  # Moderate buy signal
        # Caution for buying in high volatility
        elif vol_20d > high_volatility_threshold:
            reward -= 20  # Penalize buying in high volatility

    else:  # Holding
        # Reward for holding in strong trend
        if trend_r_squared > 0.8:
            reward += 20  # Encourage holding
        # Consider selling in weak trend or overbought conditions
        elif trend_r_squared < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 30  # Strong sell signal
        # Penalize holding in high volatility
        elif vol_5d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)