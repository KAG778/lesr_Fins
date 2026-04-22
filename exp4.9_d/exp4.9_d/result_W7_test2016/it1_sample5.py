import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # Caution in high volatility
    strong_trend_threshold = 0.8  # Strong trend indicator
    weak_trend_threshold = 0.5  # Weak trend indicator
    overbought_threshold = 0.8  # Overbought condition for Bollinger Bands
    oversold_threshold = 0.2  # Oversold condition for Bollinger Bands

    if position == 0:  # Not holding the stock
        # Conditions for BUY
        if trend_r2 > strong_trend_threshold and bb_pos < oversold_threshold:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > strong_trend_threshold and bb_pos < 0.3:  # Slightly oversold
            reward += 30  # Moderate buy signal
        elif volatility_ratio < 1.5:  # Low volatility
            reward += 20  # Low volatility encourages buying
        else:
            reward -= 10  # Neutral or weak conditions for buying
        
    else:  # Holding the stock
        # Conditions for HOLD or SELL
        if trend_r2 > strong_trend_threshold:  # Strong uptrend
            reward += 20  # Encourage holding during strong uptrend
        elif bb_pos > overbought_threshold:  # Overbought condition
            reward -= 40  # Consider selling due to overbought
        elif trend_r2 < weak_trend_threshold:  # Weak trend
            reward -= 30  # Consider selling if trend weakens
        elif volatility_ratio > high_volatility_threshold:  # Caution in extreme volatility
            reward -= 20  # Be cautious in high volatility
        
    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward