import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state

    # Extract relevant features from the enhanced state
    close_prices = s[0:20]
    position = s[150]

    # Calculate average and historical volatility
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_vol = (vol_5d + vol_20d) / 2

    # Regime features
    regime_vol = s[144]  # Volatility regime ratio
    trend_r_squared = s[145]  # R² of the trend
    bb_pos = s[149]  # Bollinger Band position

    # Initialize the reward
    reward = 0.0

    # Define thresholds
    high_vol_threshold = 2.0  # Extreme market volatility
    strong_trend_threshold = 0.8  # Strong trend R²
    overbought_bb_threshold = 0.8  # Overbought condition

    # Reward structure based on position
    if position == 0:  # Not holding
        # Identify clear BUY opportunities
        if trend_r_squared > strong_trend_threshold and bb_pos < 0.5:
            reward += 50  # Strong uptrend and not overbought
        elif regime_vol < 1.0 and np.mean(close_prices[-5:]) < np.mean(close_prices[-20:]):
            reward += 30  # Potential oversold bounce

    elif position == 1:  # Holding
        # Encourage HOLD during uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Continue holding in a strong trend
        elif bb_pos > overbought_bb_threshold or regime_vol > high_vol_threshold:
            reward -= 50  # Consider selling if overbought or in extreme volatility

    # Introduce penalties for extreme positions
    # Avoid giving rewards for extreme actions
    if regime_vol > high_vol_threshold:
        reward -= 20  # Penalty for trading in extreme conditions

    # Ensure reward stays within a reasonable range
    reward = np.clip(reward, -100, 100)

    return reward