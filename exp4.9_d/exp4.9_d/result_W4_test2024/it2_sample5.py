import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day historical volatility
    vol_20d = s[136]   # 20-day historical volatility
    vol_ratio = s[144] # Volatility regime ratio
    
    # Compute average volatility for relative thresholds
    avg_volatility = (vol_5d + vol_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility
    
    # Initialize reward
    reward = 0.0
    
    if position == 0:  # Not holding (BUY phase)
        # Strong buy signal conditions
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy opportunity
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy opportunity
        elif trend_r2 < 0.5:  # Weak trend caution
            reward -= 15  # Penalize for attempting to buy in weak trend
        if vol_ratio > high_vol_threshold:  # High volatility caution
            reward -= 20  # Caution in high volatility markets
        elif vol_ratio < low_vol_threshold:  # Low volatility encouragement
            reward += 10  # Slightly reward buying in low volatility
    
    else:  # Holding (SELL phase)
        # Encourage holding during strong trends
        if trend_r2 > 0.8:  # Strong trend
            reward += 30  # Positive reward for holding
        elif trend_r2 > 0.6 and bb_pos < 0.8:  # Moderate trend and not overbought
            reward += 10  # Mild reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong encouragement to sell
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Encourage selling to avoid losses

        # Manage risk in high volatility
        if vol_ratio > high_vol_threshold:  # High volatility caution
            reward -= 30  # Penalize for holding in high volatility
        elif vol_ratio < low_vol_threshold:  # Low volatility encouragement
            reward += 5  # Reward for holding in low volatility
    
    # Normalize the reward to fit the range [-100, 100]
    return np.clip(reward, -100, 100)