import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Position flag
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    
    # Historical Volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    vol_mean = (vol_5d + vol_20d) / 2
    high_vol_threshold = vol_mean * 2  # High volatility threshold
    low_vol_threshold = vol_mean * 0.5  # Low volatility threshold
    
    # Trend and regime features
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]    # Bollinger Band position
    price_pos = s[146] # Price position in the 20-day range
    
    # Initialize reward
    reward = 0

    # If not holding the stock
    if position == 0:
        # Strong buy conditions
        if trend_r2 > 0.8 and price_pos < 0.3 and vol_20d < low_vol_threshold:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and price_pos < 0.4:  # Moderate buy signal
            reward += 30  # Moderate buy signal
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 20  # Avoid buying in overbought conditions
        elif vol_20d > high_vol_threshold:  # High volatility caution
            reward -= 10  # Avoid buying in high volatility

    # If holding the stock
    elif position == 1:
        # Reward for holding in a strong uptrend
        if trend_r2 > 0.8:
            reward += 30  # Strong hold signal
        elif trend_r2 < 0.5 and bb_pos > 0.8:  # Weak trend and overbought
            reward -= 50  # Strong sell signal
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 20  # Consider selling
        elif bb_pos > 0.6:  # Mildly overbought
            reward -= 10  # Consider selling

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)