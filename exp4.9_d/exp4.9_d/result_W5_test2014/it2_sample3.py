import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R² of regression)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]

    # Define volatility thresholds
    high_volatility_threshold = 1.5 * volatility_20d  # High volatility for caution
    low_volatility_threshold = 0.5 * volatility_20d  # Low volatility for potential buying

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding (potential buying scenario)
        # Strong buy signal conditions
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.3:  # Moderately strong trend
            reward += 30  # Moderate buy signal
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 10  # Slight penalty for uncertainty
        # Penalize potential buying in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution on buying in high volatility

    elif position == 1:  # Holding (potential selling scenario)
        # Reward for holding during strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Positive reinforcement for holding
        # Encourage selling in weak trends or overbought conditions
        if trend_r_squared < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 40  # Strong signal to consider selling
        # Penalize holding during high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution on holding in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward