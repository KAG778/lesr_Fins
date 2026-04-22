import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    rsi_14 = s[129]  # 14-day RSI
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # Caution in high volatility
    strong_trend_threshold = 0.8  # Strong trend indicator
    weak_trend_threshold = 0.5  # Weak trend indicator
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition for Bollinger Bands and RSI
    
    if position == 0:  # Not holding (Buy Signal)
        # Strong buy signal
        if trend_r_squared > strong_trend_threshold and bb_pos < oversold_threshold and rsi_14 < 30:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal under low volatility
        elif trend_r_squared < weak_trend_threshold and bb_pos < oversold_threshold and volatility_ratio < 1.5:
            reward += 20  # Moderate buy opportunity
        else:
            reward -= 5  # Neutral signal, slight penalty to discourage buying in uncertain conditions

    else:  # Holding (Sell Signal)
        # Reward for holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Positive reward for holding in a strong trend
        # Consider selling conditions
        if bb_pos > overbought_threshold or rsi_14 > 70:  # Overbought conditions
            reward -= 50  # Strong incentive to sell
        elif trend_r_squared < weak_trend_threshold:  # Weak trend
            reward -= 30  # Encourage selling
        else:
            reward += 10  # Neutral or slight positive for holding in stable conditions
        
        # Penalize for extreme volatility
        if volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward