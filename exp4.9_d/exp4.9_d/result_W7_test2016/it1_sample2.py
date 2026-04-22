import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features
    s = enhanced_state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    rsi_14 = s[129]  # 14-day RSI
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d != 0 else 0
    
    # Initialize reward
    reward = 0
    
    # Define relative thresholds based on volatility
    high_volatility_threshold = 2.0
    low_volatility_threshold = 0.5
    
    if position == 0:  # Not holding the stock (Buy Signal)
        # Check for strong buy conditions
        if trend_r_squared > 0.8 and bb_position < 0.2 and rsi_14 < 30:
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and volatility_ratio < low_volatility_threshold:
            reward += 30  # Moderate buy opportunity
        elif bb_position < 0.2 and rsi_14 < 30:
            reward += 20  # Additional buy opportunity
        
        # Penalize buying in extreme volatility
        if volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    else:  # Holding the stock (Sell Signal)
        # Reward for holding in a strong trend
        if trend_r_squared > 0.8:
            reward += 30  # Positive reward for holding
        elif bb_position > 0.8 or rsi_14 > 70:
            reward -= 50  # Consider selling due to overbought conditions
        elif trend_r_squared < 0.5:
            reward -= 30  # Weak trend, consider selling
        else:
            reward += 10  # Neutral hold signal

        # Penalize for extreme volatility while holding
        if volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    # Normalize reward to be within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward