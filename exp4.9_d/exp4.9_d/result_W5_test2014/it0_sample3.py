import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features for clarity
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Define reward variables
    reward = 0
    
    # Calculate relative volatility thresholds
    high_volatility_ratio = 2.0
    high_trend_r_squared = 0.8
    high_bb_threshold = 0.8
    
    # Reward logic based on position
    if position == 0:  # Not holding
        # Identify strong BUY opportunities
        if trend_r_squared > high_trend_r_squared:
            reward += 50  # Strong uptrend
        if bb_position < 0.2:  # Considered oversold
            reward += 30  # Positive reward for potential bounce
    else:  # Holding
        # Encourage HOLD during uptrend
        if trend_r_squared > high_trend_r_squared:
            reward += 20  # Continue holding in a strong trend
        # Encourage SELL when trend weakens or overbought
        if trend_r_squared < 0.5 or bb_position > high_bb_threshold:
            reward += -50  # Signal to sell if trend weakens or overbought
    
    # Adjust rewards based on volatility regime
    if volatility_5d / volatility_20d > high_volatility_ratio:
        reward += -20  # Caution in high volatility markets
    
    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward