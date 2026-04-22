import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0

    # Extract features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_ratio = s[144] # Volatility regime ratio
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136] # 20-day historical volatility
    momentum = s[134]   # 10-day rate of change
    rsi_5d = s[128]     # 5-day RSI
    rsi_10d = s[129]    # 10-day RSI

    # Define thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d  # High volatility regime
    low_vol_threshold = 0.5 * volatility_20d  # Low volatility regime

    # Reward logic based on position
    if position == 0:  # Not holding
        # Positive reward for strong uptrend (high R², positive momentum)
        if trend_r2 > 0.8 and momentum > 0:
            reward += 50  # Strong buy signal

        # Reward for being oversold (RSI < 30)
        if rsi_5d < 30:
            reward += 30  # Potential bounce

        # Penalize if in high volatility regime (be cautious)
        if vol_ratio > 2:
            reward -= 20  # Caution in extreme market conditions

    elif position == 1:  # Holding
        # Positive reward for holding in a strong trend
        if trend_r2 > 0.8:
            reward += 20  # Continue holding

        # Encourage selling when trend weakens (low momentum)
        if momentum < 0:
            reward -= 30  # Consider selling

        # Consider selling if overbought (Bollinger Band position > 0.8)
        if bb_pos > 0.8:
            reward -= 40  # Selling signal

        # Penalize if in high volatility regime
        if vol_ratio > 2:
            reward -= 20  # Caution in extreme market conditions
        
    # Normalize the reward to fit the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward