import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced state
    s = enhanced_state
    close_prices = s[0:20]
    position = s[150]
    
    # Historical volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    
    # Regime features
    regime_vol = s[144]  # Volatility regime ratio
    trend_r2 = s[145]    # Trend strength (R² of regression)
    bb_pos = s[149]      # Bollinger Band position [0,1]
    
    # Reward initialization
    reward = 0.0
    
    # Define thresholds based on historical volatility
    vol_threshold = 1.5 * vol_5d  # Adjusted threshold for decision-making
    strong_trend_threshold = 0.8  # R² threshold for strong trend
    overbought_threshold = 0.8     # Bollinger Band position

    if position == 0:  # Not holding stock
        # Conditions for a BUY signal
        if trend_r2 > strong_trend_threshold and bb_pos < 0.5:  # Strong uptrend and not overbought
            reward += 50  # Strong buy opportunity
        if vol_20d < vol_threshold:  # Low volatility environment
            reward += 20  # Additional incentive for buying in low volatility
    else:  # Holding stock
        # Conditions for a HOLD signal
        if trend_r2 > strong_trend_threshold:  # Strong trend
            reward += 30  # Encourage holding
        if bb_pos > overbought_threshold:  # Overbought conditions
            reward -= 50  # Risk of price drop; consider selling
        
        # Conditions for a SELL signal
        if trend_r2 < 0.5:  # Weak trend
            reward -= 40  # Encourage selling
        if regime_vol > 2:  # High volatility
            reward -= 30  # Caution advised in extreme volatility
    
    # Normalize the reward to be within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward