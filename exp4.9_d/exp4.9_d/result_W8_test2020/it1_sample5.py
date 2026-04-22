import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    position = s[150]  # Position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Calculate relative thresholds based on volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility

    # Initialize reward
    reward = 0

    if position == 0:  # Not holding stock (BUY conditions)
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.4:  # Good trend, slightly oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Caution, consider waiting
        else:
            reward -= 5  # Small penalty for indecision

    elif position == 1:  # Holding stock (SELL or HOLD conditions)
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Reward for holding in a strong trend
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 20  # Consider selling
        elif volatility_5d > high_volatility_threshold:  # Extreme volatility
            reward -= 10  # Caution, consider selling
        else:
            reward += 5  # Mild reward for stable holding conditions

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward