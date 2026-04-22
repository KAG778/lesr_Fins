import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Constants
    MAX_REWARD = 100
    MIN_REWARD = -100
    
    # Position flag
    position = s[150]
    
    # Historical Volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    vol_mean = (vol_5d + vol_20d) / 2
    vol_threshold_high = vol_mean * 2  # High volatility threshold
    vol_threshold_low = vol_mean * 0.5  # Low volatility threshold
    
    # Trend features
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position
    price_pos = s[146]  # Price position in 20-day range
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Reward for strong buy signals
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += MAX_REWARD * 0.5  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate trend and near oversold
            reward += MAX_REWARD * 0.3  # Moderate buy signal
        elif bb_pos > 0.8:  # Overbought condition
            reward += MIN_REWARD * 0.3  # Penalty for buying when overbought
        elif vol_20d > vol_threshold_high:  # Caution in high volatility
            reward += MIN_REWARD * 0.2  # Penalty for buying in high volatility

    elif position == 1:  # Holding
        # Reward for holding during strong uptrends
        if trend_r2 > 0.8:  # Strong trend
            reward += MAX_REWARD * 0.4  # Encourage holding
        elif trend_r2 < 0.5 and bb_pos > 0.8:  # Weak trend and overbought
            reward += MIN_REWARD * 0.5  # Strong penalty for selling
        elif trend_r2 < 0.5:  # Weak trend
            reward += MIN_REWARD * 0.3  # Consider selling
        elif vol_5d > vol_threshold_high:  # High volatility while holding
            reward += MIN_REWARD * 0.4  # Consider selling in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, MIN_REWARD, MAX_REWARD)
    
    return reward