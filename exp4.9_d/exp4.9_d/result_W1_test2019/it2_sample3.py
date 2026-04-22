import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # R² of the trend
    bb_position = enhanced_state[149]  # Bollinger Band position
    regime_vol = enhanced_state[144]  # Volatility regime ratio

    # Calculate average volatility and set dynamic thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_vol_threshold = 2 * avg_volatility  # Dynamic high volatility threshold
    low_vol_threshold = avg_volatility / 2   # Dynamic low volatility threshold

    # Initialize reward
    reward = 0.0

    # Define key thresholds
    strong_trend_threshold = 0.8  # R² threshold for strong trend
    overbought_threshold = 0.8  # Bollinger Band position for overbought condition
    oversold_threshold = 0.2  # Bollinger Band position for oversold condition

    if position == 0:  # Not holding (BUY opportunities)
        # Strong buy signal conditions
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > strong_trend_threshold and bb_position < 0.5:
            reward += 30  # Moderate buy opportunity
        elif regime_vol < 1.0 and volatility_5d < low_vol_threshold:
            reward += 20  # Buy in calm market conditions
        else:
            reward -= 10  # Neutral or weak buy signal

        # Additional caution for buying in high volatility
        if regime_vol > 1.5:
            reward -= 20  # Penalty for aggressive buying in high volatility

    elif position == 1:  # Holding (SELL or HOLD opportunities)
        # Encourage holding in strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding in a strong trend
        elif trend_r_squared < 0.5 and bb_position > overbought_threshold:
            reward -= 50  # Strong sell signal in weak trend and overbought
        elif bb_position > overbought_threshold:
            reward -= 30  # Consider selling in overbought conditions
        elif regime_vol > high_vol_threshold:
            reward -= 30  # Caution against holding in high volatility
        elif trend_r_squared < 0.5:
            reward -= 20  # Weak trend, consider selling

    # Normalize reward to fit within the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward