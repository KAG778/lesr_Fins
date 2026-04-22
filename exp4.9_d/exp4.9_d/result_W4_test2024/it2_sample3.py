import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    s = enhanced_state
    position = s[150]  # Position flag (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio

    # Set relative volatility thresholds for decision-making
    avg_vol = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_vol
    low_vol_threshold = 0.5 * avg_vol

    # Initialize the reward
    reward = 0.0  

    if position == 0:  # Not holding (BUY phase)
        # Strong buy opportunity
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.7 and bb_pos < 0.4:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy signal
        elif trend_r2 < 0.5:  # Weak trend caution
            reward -= 10  # Penalize for buying in weak trend
        if vol_ratio > high_vol_threshold:  # Caution in high volatility
            reward -= 20  # Penalize for buying in extreme volatility
        elif vol_ratio < low_vol_threshold:  # Encourage action in low volatility
            reward += 10  # Small additional incentive for buying

    else:  # Holding (SELL phase)
        # Encourage holding during a strong trend
        if trend_r2 > 0.8:  
            reward += 20  # Positive reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong encouragement to sell
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Encourage selling to avoid losses
        if vol_ratio > high_vol_threshold:  # High volatility caution
            reward -= 30  # Penalize for holding in high volatility
        elif vol_ratio < low_vol_threshold:  # Encourage holding in low volatility
            reward += 10  # Small additional incentive for holding

    # Normalize the reward to be within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward