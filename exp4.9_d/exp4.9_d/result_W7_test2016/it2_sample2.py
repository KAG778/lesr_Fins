import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # Caution in high volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    weak_trend_threshold = 0.5  # Weak trend threshold
    overbought_threshold = 0.8  # Overbought condition
    oversold_threshold = 0.2  # Oversold condition

    # Decision making based on the position
    if position == 0:  # Not holding the stock (Buy signals)
        # Strong buy condition: high trend and oversold
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy signal
        # Moderate buy condition: low volatility and weak trend
        elif volatility_ratio < 1.5 and trend_r_squared < weak_trend_threshold and bb_position < oversold_threshold:
            reward += 30  # Moderate buy opportunity
        else:
            reward -= 10  # Neutral or weak signal, avoid buying in uncertain conditions

    else:  # Holding the stock (Sell signals)
        # Reward for holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong trend
        # Conditions to consider selling
        if bb_position > overbought_threshold or trend_r_squared < weak_trend_threshold:
            reward -= 50  # Negative reward for holding in overbought or weak trend
        elif volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in extreme volatility
        # Encourage selling when trend weakens
        if trend_r_squared < weak_trend_threshold:
            reward -= 30  # Strong incentive to sell if trend weakens

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward