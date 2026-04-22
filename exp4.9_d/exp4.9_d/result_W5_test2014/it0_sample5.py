import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]  # Holding flag
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend R²
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Calculate thresholds for volatility
    volatility_threshold_buy = 1.5 * volatility_5d  # Consider 1.5 times 5-day volatility for buy
    volatility_threshold_sell = 1.5 * volatility_20d  # Consider 1.5 times 20-day volatility for selling

    if position == 0:  # Not holding
        # Check for strong buying opportunity
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared < 0.5 and bb_position < 0.3:  # Weak trend but potential bounce
            reward += 20  # Moderate buy signal
        elif volatility_5d > volatility_threshold_buy:  # High volatility
            reward -= 10  # Caution on buying
        
    elif position == 1:  # Holding
        # Reward for holding in strong trends
        if trend_r_squared > 0.8:
            reward += 30  # Strong hold
        elif trend_r_squared < 0.5:  # Trend weakening
            reward -= 20  # Consider selling
        elif bb_position > 0.8:  # Overbought
            reward -= 30  # Consider selling
        elif volatility_20d > volatility_threshold_sell:  # High volatility
            reward -= 15  # Caution on holding

    # Normalize reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward