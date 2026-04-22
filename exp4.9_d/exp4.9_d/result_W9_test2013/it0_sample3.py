import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]
    volatility_5d = s[135]
    volatility_20d = s[136]
    trend_strength = s[145]
    bb_position = s[149]
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds for reward scaling
    volatility_threshold = 2 * volatility_20d  # Using 2x 20-day volatility
    trend_threshold = 0.8  # Strong trend threshold
    bb_overbought_threshold = 0.8  # Overbought condition
    
    if position == 0:  # Not holding
        # Reward for clear BUY opportunities
        if trend_strength > trend_threshold and bb_position < 0.5:  # Strong uptrend & not overbought
            reward += 50  # Strong positive reward
        elif bb_position < 0.2:  # Oversold condition
            reward += 30  # Moderate positive reward
        elif trend_strength < 0.5:  # Weak trend
            reward -= 20  # Negative reward for weak conditions
        # Extreme market caution
        if volatility_5d > volatility_threshold:
            reward -= 15  # Caution in high volatility

    else:  # Holding
        # Reward for HOLD during strong uptrend
        if trend_strength > trend_threshold:
            reward += 30  # Positive reward for holding
        # Encourage selling when trend weakens or overbought
        elif trend_strength < 0.5 or bb_position > bb_overbought_threshold:
            reward -= 50  # Strong negative reward to sell
        # Moderate caution for high volatility
        if volatility_5d > volatility_threshold:
            reward -= 10  # Additional caution penalty

    # Normalize reward to fit between [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward