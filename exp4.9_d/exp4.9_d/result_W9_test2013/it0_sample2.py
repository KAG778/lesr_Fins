import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features from the enhanced state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Reward initialization
    reward = 0.0

    # Use relative thresholds based on historical volatility
    high_volatility_threshold = 2 * volatility_20d  # Example threshold for high volatility
    low_volatility_threshold = 0.5 * volatility_20d  # Example threshold for low volatility
    
    # Reward logic based on current position
    if position == 0:  # Not holding stock
        # Conditions for a BUY
        if (trend_r2 > 0.8) and (bb_pos < 0.2):  # Strong trend and oversold
            reward += 50  # Strong BUY signal
        elif (trend_r2 > 0.6) and (bb_pos < 0.4):  # Moderate trend and relatively oversold
            reward += 30  # Moderate BUY signal
        elif (volatility_5d < low_volatility_threshold):  # Low volatility
            reward -= 10  # Caution against buying in low volatility

    elif position == 1:  # Holding stock
        # Conditions for a HOLD
        if (trend_r2 > 0.8):  # Strong trend
            reward += 20  # Positive reward for holding during strong trend
        elif (trend_r2 < 0.6) and (bb_pos > 0.8):  # Weak trend and overbought
            reward -= 50  # Strong SELL signal
        elif (bb_pos > 0.6):  # Moderately overbought
            reward -= 20  # Consider selling

    # Caution in extreme market conditions
    if volatility_20d > high_volatility_threshold:
        reward -= 10  # Be cautious in extreme volatility

    # Normalize reward to [-100, 100]
    reward = max(min(reward, 100), -100)

    return reward