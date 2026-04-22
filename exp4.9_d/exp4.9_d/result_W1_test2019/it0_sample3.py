import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extract relevant features from enhanced_state
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength R²
    bb_pos = s[149]  # Bollinger Band position [0,1]
    momentum = s[134]  # 10-day rate of change (momentum)
    
    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # 2x 20-day volatility
    low_volatility_threshold = 0.5 * volatility_20d  # 0.5x 20-day volatility
    strong_trend_threshold = 0.8  # R² threshold for strong trend
    overbought_threshold = 0.8  # Bollinger Band position for overbought
    
    if position == 0.0:  # Not holding
        # Reward for strong uptrend or oversold bounce
        if momentum > 0 and trend_r2 > strong_trend_threshold:  # Strong positive momentum
            reward += 50  # Strong buy signal
        elif momentum < 0 and bb_pos < 0.2:  # Oversold condition
            reward += 30  # Moderate buy signal
        else:
            reward -= 10  # Neutral or weak signal

    elif position == 1.0:  # Holding
        # Reward for holding in an uptrend
        if trend_r2 > strong_trend_threshold and momentum > 0:
            reward += 20  # Positive reward for holding
        # Penalty for considering sell when trend weakens
        if trend_r2 < strong_trend_threshold or (bb_pos > overbought_threshold and momentum < 0):
            reward -= 30  # Consider selling signal
        elif bb_pos > overbought_threshold:  # Overbought condition
            reward -= 20  # Negative reward for holding in overbought
        else:
            reward += 10  # Neutral holding

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward