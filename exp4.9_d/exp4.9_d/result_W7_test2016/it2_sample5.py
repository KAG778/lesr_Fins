import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features
    position = enhanced_state[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r2 = enhanced_state[145]  # Trend strength (R² of regression)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    
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

    if position == 0:  # Not holding the stock (Buy signals)
        # Strong buy signal criteria
        if trend_r2 > strong_trend_threshold and bb_position < oversold_threshold and volatility_ratio < 1.5:
            reward += 60  # Strong buy signal
        # Moderate buy signal criteria
        elif trend_r2 > weak_trend_threshold and bb_position < oversold_threshold:
            reward += 30  # Moderate buy opportunity
        # Caution signal for buying in high volatility
        elif volatility_ratio > high_volatility_threshold:
            reward -= 20  # Penalize buying in high volatility

    else:  # Holding the stock (Sell signals)
        # Reward for holding during strong trends
        if trend_r2 > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong trend
        # Conditions to consider selling
        if bb_position > overbought_threshold or trend_r2 < weak_trend_threshold or volatility_ratio > high_volatility_threshold:
            reward -= 40  # Strong sell signal due to overbought or weak trend
        else:
            reward += 10  # Neutral signal for holding

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward