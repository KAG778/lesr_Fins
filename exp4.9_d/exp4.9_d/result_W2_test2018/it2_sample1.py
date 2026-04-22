import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    
    # Historical Volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_volatility = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    # Trend and Bollinger Band features
    trend_r_squared = s[145]  # Trend strength R²
    bb_position = s[149]  # Bollinger Band position
    price_position = s[146]  # Price position in 20-day range

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_position < 0.2 and vol_20d < low_vol_threshold:  # Strong uptrend and oversold
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4:  # Moderate trend and near oversold
            reward += 30  # Moderate buy opportunity
        # Caution against buying in high volatility
        elif vol_20d > high_vol_threshold:
            reward -= 20  # Penalty for high volatility conditions
        # Avoid buying when overbought
        elif bb_position > 0.8:
            reward -= 10  # Mild penalty

    elif position == 1:  # Holding
        # Reward for holding during strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Strong hold signal
        # Consider selling if trend weakens or overbought
        elif trend_r_squared < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 40  # Strong sell signal
        # Penalize holding in high volatility
        if vol_5d > high_vol_threshold:
            reward -= 20  # Caution in high volatility markets
        # Encourage holding if trend is stable
        elif trend_r_squared >= 0.5:
            reward += 10  # Small reward for maintaining position

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)