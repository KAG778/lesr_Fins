import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state

    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility

    # Initialize reward
    reward = 0

    # Strategy when not holding (position = 0)
    if position == 0:
        # Clear BUY signals
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.5 and bb_pos < 0.3:  # Moderate trend and near bottom
            reward += 30  # Moderate buy signal
        elif trend_r_squared < 0.5 and bb_pos > 0.5:  # Weak trend, consider avoiding
            reward -= 10  # Caution in uncertain conditions
        elif volatility_5d > 1.5 * avg_volatility:  # High volatility
            reward -= 15  # Caution in high volatility regime
            
    # Strategy when holding (position = 1)
    else:
        # Clear HOLD signals
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Reward for holding
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Encourage to sell
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Consider selling
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Caution in high volatility
            
        # Incentivize selling in weakening trends
        if trend_r_squared < 0.6:  # Weakening trend
            reward -= 10  # Encourage to evaluate selling

    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward