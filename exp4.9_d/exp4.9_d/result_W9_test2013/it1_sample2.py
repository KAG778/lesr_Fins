import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in the 20-day range [0,1]
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds for decision-making
    low_vol_threshold = 0.8 * volatility_5d  # Conservative low volatility threshold
    high_vol_threshold = 1.5 * volatility_20d  # Conservative high volatility threshold
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought threshold
    oversold_threshold = 0.2  # Oversold threshold
    
    if position == 0:  # Not holding
        # Conditions for buying
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_strength > 0.6 and bb_position < 0.4:
            reward += 30  # Moderate buy opportunity

        # Penalize buying in high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 20  # Caution in high volatility

    else:  # Holding
        # Conditions for holding
        if trend_strength > strong_trend_threshold:
            reward += 20  # Reward for holding during strong trends
        elif trend_strength < 0.5 or bb_position > overbought_threshold:
            reward -= 50  # Strong sell signal if trend weakens or overbought
            reward -= 10  # Additional caution for overbought conditions

        # Encourage selling if trend is weak
        if trend_strength < 0.4:
            reward -= 20  # Penalty for holding in weak trend
        
        # Moderate penalty for high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Additional caution penalty

    # Normalize to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward