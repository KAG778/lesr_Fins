import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Initialize reward variable
    reward = 0

    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility threshold
    moderate_volatility_threshold = 1.0 * volatility_5d  # Moderate volatility threshold
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition
    
    if position == 0:  # Not holding (potential buy)
        # Reward for clear BUY opportunities
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > 0.6 and bb_position < 0.4:
            reward += 30  # Moderate buy signal
        
        # Penalty for buying in unfavorable conditions
        if bb_position > overbought_threshold:
            reward -= 20  # Discourage buying in overbought conditions
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution in high volatility environments
        elif volatility_5d > moderate_volatility_threshold:
            reward -= 5  # Mild caution in moderate volatility

    else:  # Holding (position = 1)
        # Reward for holding during a strong uptrend
        if trend_strength > strong_trend_threshold:
            reward += 20  # Positive reward for holding in strong trend
        
        # Encourage selling when trend weakens or overbought
        if trend_strength < 0.5 or bb_position > overbought_threshold:
            reward -= 50  # Strong sell signal
        elif bb_position > 0.6:
            reward -= 20  # Consider selling in moderately overbought conditions
        
        # Additional caution for high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Additional caution penalty for holding in high volatility
        elif volatility_5d > moderate_volatility_threshold:
            reward -= 5  # Mild caution in moderate volatility

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward