import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds based on volatility
    high_volatility_threshold = 1.5 * volatility_20d  # High volatility threshold
    low_volatility_threshold = 0.5 * volatility_20d  # Low volatility threshold
    strong_trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition

    if position == 0:  # Not holding (buy signals)
        # Reward for clear BUY opportunities
        if trend_r2 > strong_trend_threshold and bb_pos < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:
            reward += 30  # Moderate buy signal

        # Penalize buying in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution against buying in high volatility

    else:  # Holding (sell signals)
        # Reward for holding during a strong uptrend
        if trend_r2 > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong trend
        # Encourage selling when overbought or trend weakens
        if trend_r2 < 0.5 or bb_pos > overbought_threshold:
            reward -= 50  # Strong sell signal

        # Additional caution penalties
        if bb_pos > overbought_threshold:
            reward -= 10  # Encourage selling in overbought conditions
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Additional caution for holding in high volatility
    
    # Final normalization to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward