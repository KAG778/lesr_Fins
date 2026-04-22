import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Position flag (1.0 = holding, 0.0 = not holding)
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_pos = s[149]  # Bollinger Band position
    price_pos = s[146]  # Price position in 20-day range
    
    # Calculate volatility thresholds
    vol_mean = (vol_5d + vol_20d) / 2
    high_vol_threshold = vol_mean * 2
    low_vol_threshold = vol_mean * 0.5
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding the stock
        # Strong buy signal
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.5 and bb_pos < 0.3:  # Moderate trend and near oversold
            reward += 30  # Moderate buy opportunity
        # Caution against buying in high volatility
        elif vol_20d > high_vol_threshold:
            reward -= 20  # Penalize buying in high volatility
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 10  # Mild penalty for buying when overbought

    elif position == 1:  # Holding the stock
        # Reward for holding during strong uptrends
        if trend_r_squared > 0.8:
            reward += 30  # Continue holding signal
        # Consider selling if trend weakens or overbought
        elif trend_r_squared < 0.5 or bb_pos > 0.8:  # Weak trend or overbought
            reward -= 30  # Strong sell signal
        # Penalty for holding during high volatility
        if vol_5d > high_vol_threshold:
            reward -= 20  # Caution in high volatility
        # Small reward for maintaining position during moderate conditions
        elif trend_r_squared < 0.6 and bb_pos <= 0.6:
            reward += 10  # Small reward for maintaining position in moderate conditions

    # Normalize reward to fit within [-100, 100]
    return np.clip(reward, -100, 100)