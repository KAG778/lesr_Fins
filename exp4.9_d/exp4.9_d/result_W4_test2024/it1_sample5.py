import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
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
    
    reward = 0.0  # Initialize reward
    
    if position == 0:  # Not holding (BUY phase)
        # Strong buy opportunity
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate uptrend
            reward += 20  # Moderate buy signal
        # Caution in high volatility regimes
        if vol_ratio > high_vol_threshold:
            reward -= 20  # Penalize for buying in extreme volatility
    
    else:  # Holding (SELL/HOLD phase)
        # Encourage holding during strong trends
        if trend_r2 > 0.8:  # Strong trend
            reward += 30  # Positive reward for holding
        elif trend_r2 < 0.5 and bb_pos > 0.8:  # Weak trend and overbought
            reward -= 40  # Strong selling signal
        elif trend_r2 < 0.6:  # Weak trend
            reward -= 20  # Encourage selling to avoid losses
        
        # Caution in high volatility regimes
        if vol_ratio > high_vol_threshold:
            reward -= 30  # Penalize for holding in high volatility
    
    # Normalize the reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward