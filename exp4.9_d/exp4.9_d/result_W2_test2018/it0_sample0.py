import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced state
    s = enhanced_state
    position = s[150]
    
    # Volatility and trend indicators
    hist_volatility = np.mean(s[135:137])  # Average of 5-day and 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength R²
    bb_position = s[149]  # Bollinger Band position
    rsi_5 = s[128]  # 5-day RSI
    rsi_10 = s[129]  # 10-day RSI
    sma5 = s[120]  # 5-day SMA
    sma10 = s[121]  # 10-day SMA
    momentum = s[134]  # 10-day rate of change

    reward = 0

    if position == 0:  # Not holding
        # Reward for buying under favorable conditions
        if trend_r_squared > 0.8 and rsi_5 < 30:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and rsi_5 < 40:  # Moderate trend and mildly oversold
            reward += 20  # Moderate buy signal
        # Penalize ambiguous signals
        elif trend_r_squared < 0.4 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 20  # Avoid buying in unclear situations

    elif position == 1:  # Holding
        # Reward for holding during strong uptrends
        if trend_r_squared > 0.8:
            reward += 30  # Continue holding
        # Encourage selling if trend weakens
        elif trend_r_squared < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward += 20  # Consider selling
        # Penalize for holding in a downtrend
        elif trend_r_squared < 0.3 and momentum < 0:  # Strong downtrend
            reward -= 50  # Strong sell signal

    # Normalize reward to fit within the range of [-100, 100]
    # Here, we clamp the reward as needed
    return np.clip(reward, -100, 100)