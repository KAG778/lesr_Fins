import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    price = s[0]  # Latest closing price
    sma5 = s[120]  # 5-day SMA
    sma20 = s[122]  # 20-day SMA
    volatility_5d = s[135]  # 5-day historical volatility
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    reward = 0
    
    # Reward structure based on position
    if position == 0:  # Not holding
        # Conditions for buying
        if price > sma5 and price < sma20 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.8:  # Clear trend
            reward += 30  # Positive reward for potential buy
        if volatility_5d < 0.0458:  # Lower than training volatility
            reward += 20  # Encouragement for buying in low volatility
        
    elif position == 1:  # Holding
        # Conditions for holding
        if price > sma5 and trend_r2 > 0.8:  # Uptrend and strong trend
            reward += 30  # Reward for holding during a strong trend
        if bb_pos > 0.8:  # Overbought condition
            reward -= 50  # Penalty for holding in overbought condition
        if volatility_5d > 0.0458:  # Higher than training volatility
            reward -= 20  # Penalty for holding in high volatility

        # Conditions for selling
        if price < sma20 and trend_r2 < 0.5:  # Trend weakness
            reward += 50  # Strong sell signal
        
    # Final adjustments and normalization
    reward = np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]
    
    return reward