import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0
    
    # Extract relevant features
    position = s[150]
    r_squared = s[145]
    bb_position = s[149]
    volatility_5d = s[135]
    volatility_20d = s[136]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Define thresholds
    low_rsi_threshold = 30  # Oversold
    high_rsi_threshold = 70  # Overbought
    strong_trend_threshold = 0.8
    high_volatility_ratio = 2.0

    # When not holding (position = 0)
    if position == 0:
        # Check for a strong buy signal
        if r_squared > strong_trend_threshold and bb_position < 0.2:  # Strong trend and not overbought
            reward += 50  # Positive reward for buy opportunity
        elif bb_position > 0.8:  # Overbought condition
            reward -= 20  # Negative reward for buying in overbought conditions
        else:
            reward -= 10  # Neutral penalty for unclear signals

    # When holding (position = 1)
    elif position == 1:
        # Reward for holding in a strong trend
        if r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong uptrend
        elif r_squared < 0.5:  # Weak trend
            reward -= 20  # Negative reward for holding in a weak trend
        if bb_position > 0.8:  # Overbought condition
            reward += 30  # Encourage selling in overbought conditions
        elif bb_position < 0.2:  # Oversold condition
            reward -= 10  # Neutral penalty for holding in oversold conditions

    # Adjust for extreme volatility
    if avg_volatility > high_volatility_ratio * np.std(s[135:137]):
        reward -= 30  # Caution in extreme volatility conditions

    # Normalize reward to fit in the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward