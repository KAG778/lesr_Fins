import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state

    # Extract relevant features
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    avg_volatility = (volatility_5d + volatility_20d) / 2  # Average volatility

    # Define thresholds based on average volatility
    high_vol_threshold = 2 * avg_volatility
    low_vol_threshold = 0.5 * avg_volatility

    reward = 0

    if position == 0:  # Not holding stock
        # Conditions for potential BUY
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold condition
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_position < 0.4:  # Moderate trend and somewhat oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d < low_vol_threshold:  # Low volatility
            reward += 10  # Encourage buying in stable conditions
        else:
            reward -= 10  # Penalize buying in uncertain conditions

    else:  # Holding stock
        # Reward for holding during a strong uptrend
        if trend_r_squared > 0.8:
            reward += 20  # Positive reward for holding in a strong trend
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Encourage selling in weak trends
        if bb_position > 0.8:  # Overbought condition
            reward -= 40  # Strong sell signal when overbought
        elif bb_position > 0.7 and trend_r_squared < 0.6:  # Weak trend in overbought
            reward -= 25  # Incentivize selling

        # Penalize holding during high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 20  # Caution during high volatility

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward