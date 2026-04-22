import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract necessary features from the enhanced state
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_R2 = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in the 20-day range [0,1]
    
    # Initialize the reward variable
    reward = 0.0
    
    # Define thresholds
    low_vol_threshold = 1.0 * volatility_5d  # 1x 5-day volatility
    high_vol_threshold = 2.0 * volatility_20d  # 2x 20-day volatility
    
    # Reward structure when not holding (position = 0)
    if position == 0:
        # Consider a BUY signal
        if price_pos < 0.2 and trend_R2 > 0.8:  # Oversold with strong uptrend
            reward += 20
        elif price_pos < 0.4 and trend_R2 > 0.5:  # Moderately oversold
            reward += 10
        elif trend_R2 < 0.3:  # Weak trend
            reward -= 5  # Penalty for indecisive market
    
    # Reward structure when holding (position = 1)
    else:
        # Consider a SELL signal
        if trend_R2 > 0.8 and bb_position > 0.8:  # Overbought with strong uptrend
            reward += 15  # Positive reward for selling
        elif trend_R2 < 0.5 and bb_position > 0.6:  # Weak trend and close to overbought
            reward += 5   # Moderate reward for holding
        elif trend_R2 < 0.3:  # Weak trend
            reward -= 10  # Penalty for holding in weak trend
        
        # Encourage holding during strong uptrends
        if trend_R2 > 0.8:
            reward += 10  # Reward for staying in strong trend
    
    # Normalization to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward