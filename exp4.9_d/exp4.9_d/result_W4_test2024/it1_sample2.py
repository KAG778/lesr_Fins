import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0

    # Extract relevant features
    position = s[150]  # Current position flag
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio
    avg_volatility = (vol_5d + vol_20d) / 2  # Average volatility

    # Define thresholds
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility

    if position == 0:  # Not holding
        # Strong buy opportunity
        if trend_r2 > 0.8 and bb_pos < 0.2:  
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  
            reward += 30  # Moderate buy signal
        elif vol_ratio > 2:  # Caution in high volatility
            reward -= 20  # Reduce reward due to risk
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 10  # Penalize for considering a buy in weak trend

    elif position == 1:  # Holding
        # Encourage holding during a strong trend
        if trend_r2 > 0.8:  
            reward += 30  # Strong hold signal
        elif trend_r2 > 0.5 and bb_pos < 0.8:  
            reward += 10  # Continue holding in moderate trend
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong encouragement to sell
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Encourage selling
        if vol_ratio > 2:  # High volatility
            reward -= 20  # Caution advised

    # Normalize the reward to fit within the range [-100, 100]
    return np.clip(reward, -100, 100)