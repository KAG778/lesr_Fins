import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting features from the enhanced state
    s = enhanced_state
    position = s[150]  # Position: 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    price_pos = s[146]  # Price position in 20-day range [0, 1]
    
    # Calculate thresholds based on volatility
    volatility_threshold_high = 2 * volatility_20d
    volatility_threshold_low = 0.5 * volatility_20d
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding the stock
        # Reward for clear BUY opportunities
        if trend_r_squared > 0.8 and price_pos < 0.3:  # Strong uptrend and oversold condition
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.5 and price_pos < 0.4:  # Moderate uptrend and near oversold
            reward += 30  # Moderate buy signal
        elif volatility_20d < volatility_threshold_low:  # Low volatility
            reward += 10  # Cautious buy signal
    else:  # Holding the stock
        # Reward for HOLD during uptrend
        if trend_r_squared > 0.8:  # Clear uptrend
            reward += 20  # Encourage holding
        elif trend_r_squared < 0.5 and price_pos > 0.6:  # Weak trend and overbought
            reward -= 30  # Consider selling
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 50  # Strong sell signal
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 20  # Consider selling but not as strongly

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward