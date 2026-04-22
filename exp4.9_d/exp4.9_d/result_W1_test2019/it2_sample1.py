import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    position = enhanced_state[150]  # Position flag: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # R² of the trend
    bb_position = enhanced_state[149]  # Bollinger Band position [0,1]
    regime_vol = enhanced_state[144]  # Volatility regime ratio

    # Calculate average volatility for thresholds
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # Caution for high volatility
    low_volatility_threshold = avg_volatility / 2    # Caution for low volatility

    # Initialize reward
    reward = 0.0

    # Define key thresholds
    strong_trend_threshold = 0.8  # R² threshold for strong trend
    overbought_threshold = 0.8  # Bollinger Band position for overbought condition
    oversold_threshold = 0.2  # Bollinger Band position for oversold condition

    if position == 0:  # Not holding (BUY opportunities)
        # Reward for strong buy signals
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > strong_trend_threshold and bb_position < 0.5:
            reward += 30  # Moderate buy opportunity
        elif regime_vol < 1.0 and volatility_5d < low_volatility_threshold:
            reward += 20  # Buying in calm market conditions
        elif bb_position > 0.8:  # Avoid buying in overbought conditions
            reward -= 30  # Caution against overbought

    elif position == 1:  # Holding (SELL or HOLD opportunities)
        # Encourage holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Positive reward for holding in strong trend
        elif trend_r_squared < 0.5 and bb_position > overbought_threshold:
            reward -= 50  # Strong sell signal in weak trend and overbought
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Consider selling in weak trend
        elif volatility_20d > high_volatility_threshold:  # Extreme volatility caution
            reward -= 20  # Caution against holding in high volatility
        elif bb_position < 0.2:  # Oversold condition while holding
            reward -= 10  # Neutral penalty for holding in oversold conditions

    # Adjust for extreme trading actions based on volatility regime
    if regime_vol > 2:  # Extreme market condition
        reward -= 20  # Penalty for trading in extreme volatility
    elif regime_vol < 0.5:  # Low volatility
        reward += 10  # Reward for trades in a calm market

    # Normalize the reward to fit in the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward