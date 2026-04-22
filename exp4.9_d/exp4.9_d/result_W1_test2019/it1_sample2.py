import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0

    # Extract relevant features from the enhanced state
    position = s[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # R² of the trend
    bb_position = s[149]  # Bollinger Band position [0,1]
    regime_vol = s[144]  # Volatility regime ratio

    # Calculate average volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = avg_volatility / 2

    if position == 0:  # Not holding (Buy scenarios)
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > 0.6 and bb_position < 0.5:  # Moderate trend and not overbought
            reward += 30  # Moderate buy opportunity
        elif regime_vol < 1.0 and volatility_5d < low_volatility_threshold:  # Calm market conditions
            reward += 20  # Buying in a calm market
        elif bb_position > 0.8:  # Overbought condition
            reward -= 20  # Avoid buying in overbought conditions

    elif position == 1:  # Holding (Sell scenarios)
        if trend_r_squared > 0.8:  # Strong trend
            reward += 20  # Reward for holding in a strong trend
        elif trend_r_squared < 0.5 and bb_position > 0.8:  # Weak trend and overbought
            reward -= 40  # Strong signal to sell
        elif trend_r_squared < 0.5:  # Weak trend
            reward -= 30  # Consider selling in a weak trend
        elif regime_vol > high_volatility_threshold:  # Extreme volatility
            reward -= 30  # Caution in extreme conditions
        elif bb_position < 0.2:  # Oversold condition while holding
            reward -= 10  # Neutral penalty for holding in oversold conditions

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward