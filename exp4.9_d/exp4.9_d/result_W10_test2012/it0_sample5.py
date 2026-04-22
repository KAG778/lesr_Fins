import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    volatility = s[135]  # 10-day rate of change
    rsi_14 = s[128]  # 14-day RSI
    atr = s[137]  # 14-day Average True Range
    momentum = s[134]  # 10-day momentum
    
    reward = 0
    
    # Reward logic when NOT holding (position = 0)
    if position == 0:
        # Strong uptrend signal: encourage buying
        if r_squared > 0.8 and rsi_14 < 30:  # Strong trend and oversold
            reward += 50  # Encourage buy
        elif r_squared > 0.8 and momentum > 0:  # Strong trend with positive momentum
            reward += 30  # Encourage buy
        elif bb_position > 0.8:  # Overbought condition
            reward -= 20  # Discourage buying

    # Reward logic when holding (position = 1)
    elif position == 1:
        # Maintain position in a strong uptrend
        if r_squared > 0.8 and rsi_14 > 50:  # Strong trend with RSI indicating strength
            reward += 30  # Encourage hold
        elif r_squared < 0.5 or rsi_14 > 70 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 50  # Encourage sell
        elif volatility > 1.5 * atr:  # High volatility regime
            reward -= 30  # Caution advised, potential for sell

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward