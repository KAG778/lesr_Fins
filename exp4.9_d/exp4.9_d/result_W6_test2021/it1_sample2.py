import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    price = s[0]  # Current price

    # Calculate relative thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility

    if position == 0:  # Not holding stock
        # Buy conditions
        if trend_r_squared > 0.8 and bb_pos < 0.3:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_pos < 0.4:  # Moderate trend and near bottom
            reward += 30  # Moderate buy signal
        elif price < s[120]:  # Current price below 5-day SMA
            reward -= 10  # Neutral penalty for buying at high levels

        # Caution in high volatility regime
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Caution in extreme market conditions

    else:  # Holding stock
        # Hold conditions
        if trend_r_squared > 0.8:  # Strong trend
            reward += 30  # Reward for holding in strong trend
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Penalty for holding in weak trend
        
        # Consider selling
        if bb_pos > 0.8:  # Overbought condition
            reward -= 40  # Encourage to sell
        elif price < s[120] and trend_r_squared < 0.6:  # Weak trend and below SMA
            reward -= 20  # Suggest selling

        # Additional caution for high volatility
        if volatility_20d > high_vol_threshold:
            reward -= 15  # Caution in high volatility

    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward