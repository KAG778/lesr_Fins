import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract position and relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0,1]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    price_position = s[146]  # Price position in 20-day range [0,1]
    
    reward = 0.0
    
    if position == 0:  # Not holding
        # Reward for buying under good conditions
        if trend_r2 > 0.8 and price_position < 0.3:  # Strong uptrend and oversold
            reward += 50.0
        elif trend_r2 > 0.6 and price_position < 0.4:  # Slightly weaker conditions
            reward += 25.0
        # Caution for extreme volatility regimes
        elif volatility_5d / volatility_20d > 2:
            reward -= 20.0  # High volatility, be cautious
        else:
            reward -= 5.0  # Neutral case, slight penalty
            
    elif position == 1:  # Holding
        # Reward for holding in strong trends
        if trend_r2 > 0.8:
            reward += 30.0
        # Penalize for selling in overbought conditions
        if bb_pos > 0.8:  # Overbought conditions
            reward -= 30.0
        elif trend_r2 < 0.5:  # Weak trend, consider selling
            reward -= 20.0
        else:
            reward += 5.0  # Neutral case, slight reward for holding
            
    # Ensure reward is within the specified range
    reward = np.clip(reward, -100, 100)
    
    return reward