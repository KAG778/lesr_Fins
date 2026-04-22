import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    rsi_5 = s[128]  # 5-day RSI
    rsi_10 = s[129]  # 10-day RSI
    rsi_14 = s[130]  # 14-day RSI
    trend_r_squared = s[145]  # R² of trend
    bb_pos = s[149]  # Bollinger Band position
    vol_ratio = s[144]  # Volatility regime ratio
    hist_vol_5d = s[135]  # 5-day historical volatility
    hist_vol_20d = s[136]  # 20-day historical volatility
    
    # Set up reward variable
    reward = 0
    
    # Define thresholds
    rsi_oversold = 30  # Typical oversold level
    rsi_overbought = 70  # Typical overbought level
    high_trend_threshold = 0.8  # Strong trend threshold
    high_volatility_threshold = 2.0  # High volatility regime
    
    # Reward/Penalty for Not Holding (Buy Signal)
    if position == 0:
        if (rsi_5 < rsi_oversold) and (trend_r_squared > high_trend_threshold):
            reward += 50  # Strong buy opportunity
        elif (trend_r_squared < high_trend_threshold) and (bb_pos < 0.2):
            reward += 20  # Moderate buy opportunity
        else:
            reward += -10  # Neutral or weak signal

    # Reward/Penalty for Holding
    elif position == 1:
        if trend_r_squared > high_trend_threshold:
            reward += 30  # Continue holding during strong uptrend
        elif (rsi_5 > rsi_overbought) or (bb_pos > 0.8):
            reward += -50  # Consider selling due to overbought conditions
        elif (trend_r_squared < 0.5):
            reward += -20  # Trend weakening, consider selling
        else:
            reward += 10  # Neutral hold signal

    # Adjust for extreme volatility regime
    if vol_ratio > high_volatility_threshold:
        reward += -15  # Caution in extreme volatility

    # Clip the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward