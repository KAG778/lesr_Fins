import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features
    s = enhanced_state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # R² of trend
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 0
    
    # Initialize reward
    reward = 0.0
    
    if position == 0:  # Not holding the stock
        # Conditions for BUY
        if trend_r_squared > 0.8 and bb_position < 0.2 and volatility_ratio < 1.5:
            reward += 60  # Strong buy signal when oversold and low volatility
        elif trend_r_squared > 0.5 and bb_position < 0.3:
            reward += 30  # Moderate buy signal
        else:
            reward -= 10  # Neutral or weak conditions for BUY

    else:  # Holding the stock
        # Reward for holding during strong trends
        if trend_r_squared > 0.8:
            reward += 20  # Positive reward for holding in a strong trend
        elif bb_position > 0.8 or trend_r_squared < 0.5:
            reward -= 50  # Consider selling due to overbought conditions or weak trend
        elif volatility_ratio > 2.0:
            reward -= 20  # Be cautious in extreme volatility
        
        # Encourage selling when conditions weaken
        if trend_r_squared < 0.5:
            reward -= 30  # Strong incentive to sell if trend weakens

    # Penalize for high-frequency trading
    if position != 0:  # If currently holding and position changes
        reward -= 5  # Small penalty for holding (to discourage overtrading)

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward