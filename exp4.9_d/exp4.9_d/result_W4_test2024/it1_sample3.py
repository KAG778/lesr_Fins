import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced state
    s = enhanced_state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    
    # Historical volatility
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    vol_avg = (volatility_5d + volatility_20d) / 2
    
    # Regime features
    regime_vol = s[144]  # Volatility regime ratio
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    
    # Initialize the reward
    reward = 0.0
    
    # Define thresholds based on historical volatility
    high_vol_threshold = 2 * vol_avg
    low_vol_threshold = 0.5 * vol_avg

    if position == 0:  # Not holding stock (BUY phase)
        # Conditions for a BUY signal
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy opportunity
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate uptrend
            reward += 20  # Moderate buy opportunity
        elif regime_vol > high_vol_threshold:  # Caution in high volatility
            reward -= 20  # Caution penalty in extreme volatility
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 10  # Avoid buying in weak trend
        
    else:  # Holding stock (SELL/HOLD phase)
        # Encourage holding during strong uptrend
        if trend_r2 > 0.8:  # Strong trend
            reward += 30  # Encourage holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Consider selling due to overbought
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Encourage selling to avoid losses
            
        # Additional caution for high volatility
        if regime_vol > 2:  # High volatility
            reward -= 20  # Caution penalty

    # Normalize the reward to be within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward