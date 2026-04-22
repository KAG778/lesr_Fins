import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0

    # Extract relevant features
    price = s[0:20]  # Closing prices for the last 20 days
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    
    # Calculate recent volatility
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    average_volatility = (volatility_5d + volatility_20d) / 2
    
    # Calculate simple moving averages
    sma_5 = s[120]  # 5-day SMA
    sma_20 = s[122]  # 20-day SMA
    
    # Calculate trend strength (R²)
    trend_r2 = s[145]

    # Calculate Bollinger Band position
    bb_pos = s[149]

    # Decision-making based on the position
    if position == 0.0:  # Not holding
        # Reward for potential BUY opportunities
        if price[-1] > sma_5 and price[-1] > sma_20:  # Strong uptrend
            reward += 50  # Strong signal to buy
        
        if price[-1] < sma_20 and price[-1] < sma_5:  # Oversold condition
            reward += 30  # Opportunity to buy
    else:  # Holding
        # Reward for maintaining position during strong uptrend
        if price[-1] > sma_5 and price[-1] > sma_20 and trend_r2 > 0.8:
            reward += 40  # Continue holding in an uptrend

        # Check for conditions to sell
        if bb_pos > 0.8:  # Overbought condition
            reward -= 50  # Consider selling

        if trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Signal to sell

    # Penalize based on volatility regime
    if average_volatility > 2 * volatility_20d:  # High volatility regime
        reward -= 20  # Be cautious

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward