import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced state
    s = enhanced_state
    position = s[150]  # Position flag: 1.0 = holding stock, 0.0 = not holding
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position
    price_pos = s[146]  # Price position in 20-day range [0, 1]
    
    # Calculate thresholds based on volatility
    volatility_high_threshold = 2 * np.mean([vol_5d, vol_20d])
    volatility_low_threshold = 0.5 * np.mean([vol_5d, vol_20d])
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Strong buy signal: clear trend and oversold
        if trend_r_squared > 0.8 and price_pos < 0.2 and vol_20d < volatility_low_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > 0.6 and price_pos < 0.4:
            reward += 30  # Moderate buy opportunity
        # Caution against high volatility
        elif vol_20d > volatility_high_threshold:
            reward -= 20  # Avoid buying in high volatility
        # Overbought condition penalty
        elif bb_pos > 0.8:
            reward -= 30  # Avoid buying in overbought conditions

    elif position == 1:  # Holding
        # Reward for holding in a strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Encourage holding
        # Consider selling: weak trend or overbought
        elif trend_r_squared < 0.5 or bb_pos > 0.8:
            reward -= 50  # Strong sell signal
        # Caution in high volatility
        elif vol_5d > volatility_high_threshold:
            reward -= 30  # Caution in high volatility
        # Minor reward for maintaining position in a mildly bullish trend
        elif trend_r_squared > 0.5 and bb_pos <= 0.6:
            reward += 10  # Small reward for maintaining position

    # Normalize reward to fit within the range of [-100, 100]
    return np.clip(reward, -100, 100)