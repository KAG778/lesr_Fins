import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced_state
    s = enhanced_state
    position = s[150]  # Position flag (1.0 = holding stock, 0.0 = not holding)
    
    # Historical volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_volatility = (vol_5d + vol_20d) / 2  # Average volatility for thresholding

    # Regime features
    regime_vol = s[144]  # Volatility regime ratio
    trend_r2 = s[145]    # Trend strength (R² of regression)
    bb_pos = s[149]      # Bollinger Band position [0,1]

    # Reward initialization
    reward = 0.0

    # Define thresholds based on volatility
    high_vol_threshold = 2 * avg_volatility  # High volatility threshold
    low_vol_threshold = 0.5 * avg_volatility   # Low volatility threshold

    # Buying phase (not holding)
    if position == 0:  
        # Strong buy signal
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy opportunity
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate trend and slightly oversold
            reward += 30  # Moderate buy opportunity
        
        # Risk management
        if regime_vol > high_vol_threshold:  # High volatility caution
            reward -= 20  # Caution in high volatility markets
        elif regime_vol < low_vol_threshold:  # Low volatility encouragement
            reward += 10  # Small additional incentive

    # Holding phase (currently holding)
    elif position == 1:  
        # Holding during a strong trend
        if trend_r2 > 0.8:  # Continue holding in a strong trend
            reward += 30  # Positive reward for holding
        elif trend_r2 > 0.5 and bb_pos < 0.8:  # Moderate trend and not overbought
            reward += 10  # Mild hold signal
        
        # Selling conditions
        if bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Strong encouragement to sell
        elif trend_r2 < 0.5:  # Weakening trend
            reward -= 30  # Strong encouragement to sell
        
        # Risk management
        if regime_vol > high_vol_threshold:  # High volatility caution
            reward -= 20  # Caution in high volatility

    # Normalize the reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward