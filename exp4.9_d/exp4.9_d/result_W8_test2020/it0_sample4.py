import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Position flag (1.0 = holding, 0.0 = not holding)
    position = s[150]
    
    # Historical volatility and other metrics
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Calculate thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d  # High volatility condition
    low_vol_threshold = 0.5 * volatility_20d  # Low volatility condition
    
    # Reward logic based on position and market conditions
    if position == 0:  # Not holding
        # Check for buy conditions
        if trend_r2 > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Positive reward for Buy
        elif trend_r2 < 0.5 and volatility_5d > high_vol_threshold:  # Weak trend and high volatility
            reward -= 20  # Negative reward, be cautious
        else:
            reward -= 5  # Small penalty for indecision

    elif position == 1:  # Holding
        # Check for hold or sell conditions
        if trend_r2 > 0.8:  # Strong uptrend
            reward += 20  # Positive reward for holding
        elif trend_r2 < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward += 30  # Encourage selling
        elif volatility_5d > high_vol_threshold:  # High volatility
            reward -= 10  # Caution when holding in high volatility
        else:
            reward -= 5  # Small penalty for indecision

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward