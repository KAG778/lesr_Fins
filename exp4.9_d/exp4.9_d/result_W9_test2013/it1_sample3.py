import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in the 20-day range [0,1]
    
    # Define reward variable
    reward = 0
    
    # Define thresholds based on historical volatility
    low_volatility_threshold = 0.5 * volatility_20d  # 50% of 20-day volatility
    high_volatility_threshold = 2.0 * volatility_20d  # 200% of 20-day volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition
    
    if position == 0:  # Not holding (potential buy)
        # Conditions for buying
        if trend_r2 > strong_trend_threshold and bb_pos < oversold_threshold:  # Strong uptrend & oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Moderate trend & relatively oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # Caution in high volatility
            reward -= 20  # Penalty for buying in high volatility
        elif trend_r2 < 0.4:  # Weak trend
            reward -= 15  # Penalty for indecisive market

    else:  # Holding (position = 1.0)
        # Reward for holding during strong uptrend
        if trend_r2 > strong_trend_threshold:
            reward += 20  # Positive reward for holding
        elif trend_r2 < 0.5 or bb_pos > overbought_threshold:  # Weak trend or overbought
            reward -= 50  # Strong sell signal
        elif volatility_5d > high_volatility_threshold:  # Caution in high volatility
            reward -= 10  # Additional caution penalty
        elif trend_r2 < 0.6 and bb_pos > 0.6:  # Moderately overbought condition
            reward -= 20  # Consider selling

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward