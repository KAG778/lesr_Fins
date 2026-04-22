import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price_position = s[146]  # Price position in 20-day range [0,1]
    
    # Calculate thresholds based on historical volatility
    high_volatility_threshold = 2 * volatility_20d
    low_volatility_threshold = 0.5 * volatility_20d
    
    # Initialize reward
    reward = 0.0
    
    # Reward structure when not holding (position = 0)
    if position == 0:
        # BUY signal logic
        if trend_r_squared > 0.8 and price_position < 0.3:  # Strong uptrend and oversold condition
            reward += 50  # Strong BUY signal
        elif bb_position < 0.2:  # Very low Bollinger Band position
            reward += 20  # Consider buying on potential bounce
        
    # Reward structure when holding (position = 1)
    elif position == 1:
        # HOLD signal logic
        if trend_r_squared > 0.8:  # Strong trend
            reward += 30  # Continue holding
        elif bb_position > 0.8:  # Overbought condition
            reward -= 40  # Consider selling
        
        # Consider selling if volatility is high and trend weakens
        if volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
        elif trend_r_squared < 0.5:  # Weakening trend
            reward -= 50  # Consider selling, trend is reversing
    
    # Normalize the reward to be in the range of [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward