import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Initialize reward
    reward = 0.0
    
    if position == 0:
        # Not holding the stock
        # Conditions for BUY
        if trend_r2 > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong BUY signal
        elif volatility_ratio < 1.5 and trend_r2 < 0.5:  # Low volatility and weak trend
            reward += 20  # Moderate BUY signal
        else:
            reward -= 10  # Neutral or weak conditions for BUY
        
    else:  # Holding the stock
        # Conditions for HOLD or SELL
        if trend_r2 > 0.8:  # Strong trend
            reward += 30  # Positive reward for holding in a strong trend
        elif bb_position > 0.8:  # Overbought condition
            reward -= 50  # Consider selling, negative reward for holding
        elif volatility_ratio > 2:  # Extreme volatility
            reward -= 20  # Be cautious, negative reward for holding
        
        # Additional negative reward for weak trend while holding
        if trend_r2 < 0.5:
            reward -= 15  # Weak trend, consider selling
    
    # Normalize reward to the range [-100, 100]
    # Clamp the reward to ensure it stays within the specified range
    reward = np.clip(reward, -100, 100)
    
    return reward