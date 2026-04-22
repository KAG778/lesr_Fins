import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    momentum = s[134]  # 10-day rate of change
    sma5 = s[120]  # 5-day SMA
    sma10 = s[121]  # 10-day SMA
    sma20 = s[122]  # 20-day SMA
    
    # Calculate thresholds based on volatility
    volatility_threshold_1 = 1.5 * volatility_5d  # Low volatility threshold
    volatility_threshold_2 = 2.0 * volatility_5d  # High volatility threshold
    high_volatility = volatility_20d > volatility_threshold_2
    
    # Initialize reward
    reward = 0.0
    
    # Strategy when not holding
    if position == 0:
        # Check for clear BUY opportunity
        if momentum > 0 and trend_r2 > 0.8:  # Strong uptrend
            reward += 50  # Positive reward for a strong upward trend
        elif bb_pos < 0.2:  # Oversold condition
            reward += 30  # Positive reward for oversold bounce
        else:
            reward -= 10  # Slight penalty for not acting in uncertain conditions

    # Strategy when holding
    else:
        # Check for HOLD opportunity
        if trend_r2 > 0.8 and momentum > 0:  # Strong uptrend
            reward += 20  # Positive reward for holding during uptrend
        elif bb_pos > 0.8:  # Overbought condition
            reward += -30  # Penalty for holding in overbought condition
        elif trend_r2 < 0.5:  # Weak trend
            reward += -20  # Penalty for holding in weak trend
        else:
            reward += 10  # Small positive for holding in uncertain conditions

        # Incentivize selling when trend weakens
        if trend_r2 < 0.6 and momentum < 0:  # Weakening trend
            reward += 40  # Positive reward for selling in a weakening trend

    # Adjust reward for high volatility
    if high_volatility:
        reward -= 20  # Caution in extreme markets

    # Normalize reward to be in [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward