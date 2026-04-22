import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative volatility threshold
    volatility_threshold = 2 * volatility_20d

    # Initialize reward
    reward = 0.0

    if position == 0.0:  # Not holding
        # Conditions for buying
        if (trend_r2 > 0.8) and (bb_pos < 0.2):  # Strong uptrend and oversold condition
            reward += 50  # Strong buy signal
        elif (bb_pos < 0.3):  # Moderate oversold condition
            reward += 20  # Potential buy signal
        elif (volatility_5d < volatility_threshold):  # Low volatility environment
            reward += 10  # Cautious buy signal
        else:
            reward -= 10  # Non-ideal conditions for buying

    elif position == 1.0:  # Holding
        # Conditions for holding or selling
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Encourage holding in a strong trend
        elif (bb_pos > 0.8):  # Overbought condition
            reward += 30  # Encourage selling in overbought conditions
        elif (volatility_5d > volatility_threshold):  # High volatility environment
            reward -= 20  # Caution against holding in high volatility
        else:
            reward -= 5  # Mild penalty for holding in uncertain conditions

    # Normalizing reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward