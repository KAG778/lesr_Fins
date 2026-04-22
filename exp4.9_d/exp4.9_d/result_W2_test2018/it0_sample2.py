import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0

    # Constants for reward scaling
    volatility_threshold = np.mean(s[135:137])  # Average of historical volatility
    trend_strength_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Bollinger Band threshold for overbought condition

    # Check the current position
    if s[150] == 0:  # Not holding
        # Reward for a clear BUY opportunity
        if s[145] > trend_strength_threshold and s[149] < 0.2:  # Strong trend and not overbought
            reward += 50  # Strong buy signal
        elif s[145] > trend_strength_threshold and s[149] >= 0.2 and s[149] < 0.5:  # Moderate buy signal
            reward += 25  # Moderate buy signal
        elif s[135] > 2 * volatility_threshold:  # Extreme volatility
            reward -= 10  # Caution, avoid buying in extreme volatility
            
    elif s[150] == 1:  # Holding
        # Reward for holding during an uptrend
        if s[145] > trend_strength_threshold:
            reward += 20  # Reward for holding in a strong trend
        elif s[145] < trend_strength_threshold and s[149] > overbought_threshold:  # Potential sell signal
            reward += 10  # Consider selling
            # Penalize for holding in an overbought condition
            reward -= 15  
        elif s[135] > 2 * volatility_threshold:  # Extreme volatility while holding
            reward -= 10  # Caution, consider selling

    # Ensure reward is clamped within [-100, 100]
    return np.clip(reward, -100, 100)