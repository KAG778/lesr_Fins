import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    low_vol_threshold = 0.5 * avg_volatility
    high_vol_threshold = 2.0 * avg_volatility
    
    reward = 0
    
    if position == 0:  # Not holding stock (buy signals)
        # Strong buy when in strong uptrend and oversold
        if trend_r_squared > 0.8 and bb_pos < 0.2:
            reward += 50  # Strong buy signal
        # Moderate buy when trend is moderate and oversold
        elif trend_r_squared > 0.5 and bb_pos < 0.3:
            reward += 30  # Potential buy signal
        # Caution in high volatility
        elif volatility_5d > high_vol_threshold:
            reward -= 15  # Penalty for buying in high volatility
        # Neutral situation
        else:
            reward -= 5  # Slight penalty for unclear conditions

    else:  # Holding stock (sell signals)
        # Reward for holding in strong trend
        if trend_r_squared > 0.8:
            reward += 20  # Positive reward for holding
        # Consider selling in overbought conditions
        if bb_pos > 0.8:
            reward -= 40  # Strong penalty for holding in overbought
        # Encourage selling in weak trend
        elif trend_r_squared < 0.5:
            reward -= 30  # Penalty for selling in weak trend
        # Neutral hold
        else:
            reward += 10  # Small positive for holding in uncertain conditions

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward