import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength
    bb_position = s[149]  # Bollinger Band position
    sma5 = s[120]  # 5-day SMA
    sma20 = s[122]  # 20-day SMA
    current_price = s[0]  # Most recent closing price
    rsi_14 = s[129]  # 14-day RSI
    
    # Calculate thresholds based on volatility
    high_vol_thresh = 2 * (volatility_5d + volatility_20d) / 2  # Average volatility threshold
    overbought_threshold = 0.8  # Overbought condition for BB position
    oversold_threshold = 0.2  # Oversold condition for RSI
    
    if position == 0:  # Not holding stock
        # Reward for clear BUY opportunities
        if (current_price > sma5) and (trend_r_squared > 0.8) and (rsi_14 < oversold_threshold):
            reward += 50  # Strong buy signal
        elif (current_price < sma5) and (trend_r_squared < 0.5) and (rsi_14 > 1 - oversold_threshold):
            reward -= 20  # Avoid buying in a downtrend
    elif position == 1:  # Holding stock
        # Reward for HOLD during uptrend
        if (current_price > sma20) and (trend_r_squared > 0.8):
            reward += 30  # Strong hold signal
        # Encourage SELL when trend weakens or overbought
        if (trend_r_squared < 0.5) or (bb_position > overbought_threshold):
            reward -= 40  # Consider selling

    # Adjust rewards based on extreme volatility
    if volatility_5d > high_vol_thresh:
        reward -= 30  # Be cautious in high volatility regimes
    
    # Clamping reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward