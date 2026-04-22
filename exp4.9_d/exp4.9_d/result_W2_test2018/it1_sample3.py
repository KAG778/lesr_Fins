import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    position = s[150]  # Position flag: 1.0 = holding stock, 0.0 = not holding
    
    # Extract volatility and trend indicators
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    vol_mean = (vol_5d + vol_20d) / 2
    volatility_threshold_high = 2 * vol_mean  # High volatility threshold
    volatility_threshold_low = 0.5 * vol_mean  # Low volatility threshold
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    price_pos = s[146]  # Price position in 20-day range [0, 1]
    
    # Initialize reward
    reward = 0
    
    if position == 0:  # Not holding the stock
        # Strong buy signal
        if trend_r_squared > 0.8 and price_pos < 0.3 and vol_20d < volatility_threshold_low:  
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.6 and price_pos < 0.4 and vol_20d < volatility_threshold_low:
            reward += 30  # Moderate buy opportunity
        # Caution against high volatility
        elif vol_20d > volatility_threshold_high:
            reward -= 20  # Avoid buying in high volatility
    
    elif position == 1:  # Holding the stock
        # Encourage holding during clear uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Strong hold signal
        # Consider selling if overbought
        elif bb_pos > 0.8:
            reward -= 50  # Strong sell signal
        # Weak trend or volatility increase
        elif trend_r_squared < 0.5 or vol_20d > volatility_threshold_high:
            reward -= 30  # Consider selling due to trend weakness or high volatility
        else:
            reward += 10  # Small reward for maintaining position in a strong trend

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)