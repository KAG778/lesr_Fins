import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features
    position = s[150]
    close_prices = s[0:20]
    sma5 = s[120]
    sma10 = s[121]
    sma20 = s[123]
    r_squared = s[145]
    bb_pos = s[149]
    volatility_5d = s[135]
    volatility_20d = s[136]
    
    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Determine thresholds
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Define reward based on position
    if position == 0:  # Not holding
        # Check for clear buy signals
        if close_prices[-1] > sma5 and close_prices[-1] > sma10 and close_prices[-1] > sma20:
            if r_squared > 0.8:  # Strong trend
                reward += 50  # Strong buy signal
            elif bb_pos < 0.2:  # Oversold condition
                reward += 30  # Buy opportunity
            else:
                reward -= 10  # Neutral or weak signal
        else:
            reward -= 5  # Not a great time to buy
        
    else:  # Holding
        # Check for hold or sell signals
        if close_prices[-1] < sma20 and r_squared < 0.5:  # Weak trend
            reward -= 40  # Suggest to sell
        elif close_prices[-1] > sma5 and r_squared > 0.8:  # Strong trend, holding is good
            reward += 20  # Hold signal
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 30  # Consider selling
        else:
            reward += 10  # Neutral hold
        
    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward