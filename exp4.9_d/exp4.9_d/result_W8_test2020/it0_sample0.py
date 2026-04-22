import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    close_prices = s[0:20]
    current_price = close_prices[-1]
    position = s[150]  # Current position (1 = holding, 0 = not holding)

    # Calculate moving averages and trends
    sma5 = s[120]
    sma10 = s[121]
    sma20 = s[123]
    trend_r2 = s[145]
    bb_pos = s[149]
    
    # Calculate volatility measures
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    regime_vol = s[144]  # Volatility regime ratio

    # Set the reward
    reward = 0.0

    # Conditions for action rewards
    if position == 0:  # Not holding
        # Conditions indicating a strong BUY opportunity
        if (current_price > sma5 and current_price > sma10) and (trend_r2 > 0.8) and (bb_pos < 0.2):
            reward = 50  # Strong buy signal
        elif (current_price < sma20) and (bb_pos < 0.2):
            reward = 30  # Buying opportunity (oversold)
        else:
            reward = -10  # Neutral or weak signal

    elif position == 1:  # Holding
        # Conditions for maintaining the position
        if (current_price > sma5) and (current_price > sma10) and (trend_r2 > 0.8):
            reward = 20  # Strong hold signal
        # Conditions indicating it might be time to sell
        elif (current_price < sma20) or (bb_pos > 0.8):
            reward = -50  # Consider selling (overbought or trend weakening)
        else:
            reward = 10  # Continue holding (neutral)

    # Adjust reward based on market regime and volatility
    if regime_vol > 2:  # Extreme market conditions
        reward *= 0.5  # Reduce reward to be cautious
    elif regime_vol < 0.5:  # Low volatility regime
        reward *= 1.5  # Increase reward for aggressive trading

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward