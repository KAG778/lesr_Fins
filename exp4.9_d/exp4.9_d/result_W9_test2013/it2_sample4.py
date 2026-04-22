import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in the 20-day range [0,1]
    
    # Initialize reward variable
    reward = 0

    # Define thresholds for decision-making
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility regime
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold

    # Reward logic based on the position
    if position == 0:  # Not holding (potential buy)
        # Strong buy opportunity
        if trend_strength > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        # Moderate buy opportunity
        elif trend_strength > 0.6 and bb_position < 0.4:
            reward += 30  # Moderate buy signal
        # Penalize for buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution in high volatility

    else:  # Holding (position = 1)
        # Reward for holding during strong uptrend
        if trend_strength > strong_trend_threshold:
            reward += 20  # Positive reward for holding
        # Strong sell signal if overbought or trend weakens
        if bb_position > overbought_threshold or trend_strength < 0.5:
            reward -= 30  # Strong incentive to sell
        # Penalize holding in weak trends
        if trend_strength < 0.4:
            reward -= 20  # Penalty for holding in weak trend
        # Caution for selling in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Additional caution penalty for high volatility

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward