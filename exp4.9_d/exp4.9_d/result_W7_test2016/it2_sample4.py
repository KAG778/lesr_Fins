import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    s = enhanced_state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # High volatility caution
    strong_trend_threshold = 0.8  # Strong trend threshold
    weak_trend_threshold = 0.5  # Weak trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition

    # Initialize reward
    reward = 0.0

    # Differentiate actions based on position
    if position == 0:  # Not holding (BUY signals)
        # Strong buy opportunity: strong trend and oversold condition
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        # Moderate buy opportunity: low volatility and weak trend
        elif trend_r_squared < weak_trend_threshold and bb_position < oversold_threshold and volatility_ratio < 1.5:
            reward += 20  # Moderate buy opportunity under low volatility
        else:
            reward -= 10  # Neutral or weak signal, avoid buying in uncertain conditions

    else:  # Holding (SELL signals)
        # Reward for holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Positive reward for holding in a strong trend
        # Penalize for overbought conditions or weak trend
        if bb_position > overbought_threshold or trend_r_squared < weak_trend_threshold:
            reward -= 50  # Strong sell signal due to overbought or weakening trend
        elif volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility

    # Normalize reward to be within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward