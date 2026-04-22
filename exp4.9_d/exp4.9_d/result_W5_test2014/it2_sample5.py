import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define thresholds
    high_volatility_threshold = 1.5 * volatility_20d  # Caution for buying/selling
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Oversold condition for BB
    overbought_threshold = 0.8  # Overbought condition for BB

    # Initialize reward
    reward = 0.0

    # Reward structure when not holding (position = 0)
    if position == 0:
        # Strong buy signal
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif bb_position < oversold_threshold:
            reward += 30  # Moderate buy opportunity
        
        # Penalize for buying in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    # Reward structure when holding (position = 1)
    elif position == 1:
        # Reward for holding in a strong trend
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Strong hold signal
        # Encourage selling if overbought or trend weakens
        if bb_position > overbought_threshold or trend_r_squared < 0.5:
            reward -= 40  # Strong signal to sell
        
        # Penalize holding in high volatility
        if volatility_20d > high_volatility_threshold:
            reward -= 15  # Caution in high volatility

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward