import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]
    volatility_5d = enhanced_state[135]
    volatility_20d = enhanced_state[136]
    trend_r_squared = enhanced_state[145]
    bb_position = enhanced_state[149]
    rsi_5d = enhanced_state[128]
    momentum = enhanced_state[134]
    
    # Calculate thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d  # High volatility condition
    low_vol_threshold = 0.5 * volatility_20d  # Low volatility condition
    
    reward = 0

    if position == 0:  # Not holding
        # Reward for clear buy opportunities
        if trend_r_squared > 0.8 and momentum > 0:  # Strong uptrend
            reward += 50  # Strong buy signal
        elif rsi_5d < 30:  # Oversold condition
            reward += 30  # Potential buy opportunity
        elif volatility_5d < low_vol_threshold:  # Low volatility
            reward += 10  # Consider buying in a calm market
        elif volatility_5d > high_vol_threshold:  # High volatility
            reward -= 10  # Be cautious about buying

    elif position == 1:  # Holding
        # Reward for holding during uptrend
        if trend_r_squared > 0.8 and momentum > 0:  # Strong uptrend
            reward += 20  # Reward for holding
        elif trend_r_squared < 0.5 or momentum < 0:  # Weak trend
            reward -= 20  # Consider selling
        elif bb_position > 0.8:  # Overbought condition
            reward -= 15  # Consider selling
        elif volatility_5d > high_vol_threshold:  # High volatility
            reward -= 5  # Be cautious about holding
        
    # Normalize reward to fit within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward