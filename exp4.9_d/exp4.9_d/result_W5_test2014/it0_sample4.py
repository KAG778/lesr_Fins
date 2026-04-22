import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced_state
    position = s[150]  # Current position (holding or not)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price = s[0]  # Current price (latest closing price)
    
    # Define thresholds
    vol_threshold = 2 * volatility_20d  # High volatility threshold
    trend_threshold = 0.8  # Strong trend threshold
    overbought_threshold = 0.8  # Overbought condition for BB position

    reward = 0

    if position == 0:  # Not holding the stock
        if trend_r_squared > trend_threshold and s[126] > 0:  # Confirmed uptrend
            reward += 50  # Positive reward for strong BUY opportunity
        elif bb_position < 0.2:  # Oversold condition
            reward += 30  # Positive reward for potential bounce
        if volatility_20d > vol_threshold:  # Be cautious in high volatility
            reward -= 20  # Penalize for potential buy in extreme volatility

    elif position == 1:  # Holding the stock
        if trend_r_squared > trend_threshold:  # Confirmed uptrend
            reward += 20  # Positive reward for holding in uptrend
        if bb_position > overbought_threshold:  # Overbought condition
            reward += 30  # Positive reward for considering to sell
        if trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Penalty for holding in weak trend
        if volatility_20d > vol_threshold:  # Be cautious in high volatility
            reward -= 20  # Penalize for holding in extreme volatility

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward