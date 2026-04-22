import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    close_prices = s[0:20]
    position = s[150]
    
    # Calculate volatility metrics
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    
    # Regime features
    regime_volatility_ratio = s[144]  # Volatility regime ratio
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]

    # Reward initialization
    reward = 0
    
    # Define thresholds based on historical volatility
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility

    # Evaluate the reward based on the position
    if position == 0:  # Not holding
        # Strong buy opportunity
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Positive reward for clear buy opportunity
        elif regime_volatility_ratio > 2:  # Extreme volatility
            reward -= 20  # Caution in high volatility markets
        else:
            reward += 10  # Small positive reward for potential buy opportunities

    elif position == 1:  # Holding
        # Encourage holding during uptrend
        if trend_r_squared > 0.8:  # Clear trend
            reward += 20  # Positive reward for holding
        elif bb_position > 0.8:  # Overbought condition
            reward -= 30  # Negative reward to consider selling
        elif close_prices[-1] < close_prices[-2]:  # Price drop
            reward -= 10  # Small negative reward for price decline

    # Normalize the reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward