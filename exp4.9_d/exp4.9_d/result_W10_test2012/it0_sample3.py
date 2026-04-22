import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]
    
    # Extract relevant features
    price = s[0]  # latest closing price
    sma5 = s[120]
    sma20 = s[122]
    rsi14 = s[129]
    trend_r2 = s[145]
    bb_pos = s[149]
    volatility_5d = s[135]
    volatility_20d = s[136]
    
    reward = 0.0

    # 1. If not holding (position = 0)
    if position == 0:
        # Check for buy signals
        if price < sma20 and rsi14 < 30:
            reward += 50  # Strong buy signal (oversold condition)
        elif trend_r2 > 0.8 and price > sma5:
            reward += 30  # Confirmed uptrend
        
    # 2. If holding (position = 1)
    elif position == 1:
        # Check for sell signals
        if bb_pos > 0.8:
            reward += 40  # Overbought condition, consider selling
        elif trend_r2 < 0.5:
            reward += 30  # Weakening trend, consider selling
        else:
            reward += 20  # Hold during strong uptrend

    # Adjust rewards based on volatility
    if volatility_5d > 2 * volatility_20d:
        reward -= 10  # Penalize actions in extreme volatility
    
    # Ensure rewards are capped within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward