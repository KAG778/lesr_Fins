import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Position flag
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    
    # Volatility features
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_vol = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_vol  # High volatility threshold
    low_vol_threshold = 0.5 * avg_vol  # Low volatility threshold
    
    # Trend and Bollinger Band features
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_pos = s[149]  # Bollinger Band position
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding the stock
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.6 and bb_pos < 0.3:  # Moderate trend and near oversold
            reward += 30  # Moderate buy opportunity
        # Penalty for buying in overbought or high volatility conditions
        elif bb_pos > 0.8 or vol_20d > high_vol_threshold:
            reward -= 20  # Caution against buying in unfavorable conditions

    elif position == 1:  # Holding the stock
        # Reward for holding during a strong trend
        if trend_r_squared > 0.8:
            reward += 30  # Encourage holding
        # Consider selling if trend weakens or overbought
        elif trend_r_squared < 0.5 or bb_pos > 0.8:  # Weak trend or overbought
            reward -= 30  # Strong sell signal
        # Penalize holding in high volatility
        if vol_5d > high_vol_threshold:
            reward -= 20  # Caution required in high volatility

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)