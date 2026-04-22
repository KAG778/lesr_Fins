import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features
    position = s[150]
    
    # Historical volatility
    historical_volatility_5d = s[135]
    historical_volatility_20d = s[136]
    
    # Regime features
    vol_ratio = s[144]  # Volatility regime ratio
    trend_r2 = s[145]   # Trend strength R²
    bb_pos = s[149]     # Bollinger Band position
    
    # Reward initialization
    reward = 0.0
    
    # Setting thresholds
    volatility_threshold = 2 * historical_volatility_20d  # Relative threshold for caution
    trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition
    
    if position == 0:  # Not holding stock
        # Conditions for buying
        if trend_r2 > trend_threshold and bb_pos < 0.3:  # Strong uptrend and not overbought
            reward += 50  # Strong buy signal
        elif trend_r2 <= trend_threshold and bb_pos < 0.5:  # Weak trend but still low BB position
            reward += 20  # Moderate buy signal
            
    elif position == 1:  # Holding stock
        # Conditions for holding
        if trend_r2 > trend_threshold:  # Strong trend
            reward += 30  # Positive reward for holding in a strong trend
        elif trend_r2 <= trend_threshold and bb_pos > overbought_threshold:  # Weak trend and overbought
            reward -= 30  # Negative reward for holding in a weak trend and overbought situation
        elif trend_r2 <= trend_threshold:  # Weak trend
            reward -= 50  # Strong signal to sell in a weak trend
    
    # Adjust reward based on volatility regime
    if vol_ratio > 2:  # Extreme market condition
        reward -= 20  # Caution in extreme volatility
    elif vol_ratio < 0.5:  # Low volatility
        reward += 10  # Reward for trades in a calm market

    # Normalize the reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward