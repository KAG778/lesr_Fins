import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    rsi_5 = s[128]     # 5-day RSI
    rsi_10 = s[129]    # 10-day RSI
    trend_r2 = s[145]  # Trend strength (R²)
    bb_pos = s[149]    # Bollinger Band position [0, 1]
    vol_5d = s[135]    # 5-day volatility
    vol_20d = s[136]   # 20-day volatility
    vol_ratio = s[144] # Volatility regime ratio
    
    # Define the reward
    reward = 0
    
    # Thresholds (these can be tuned)
    rsi_oversold = 30
    rsi_overbought = 70
    high_volatility_threshold = 2.0
    strong_trend_threshold = 0.8
    high_bb_threshold = 0.8
    
    # If not holding position
    if position == 0:
        # Reward for a strong BUY opportunity
        if (rsi_5 < rsi_oversold and trend_r2 > strong_trend_threshold):
            reward += 50  # Strong BUY signal
        elif (rsi_10 < rsi_oversold and trend_r2 > strong_trend_threshold):
            reward += 30  # Moderate BUY signal
        else:
            reward -= 10  # No clear BUY opportunity
    
    # If holding position
    else:
        # Reward for HOLD during uptrend
        if (trend_r2 > strong_trend_threshold):
            reward += 20  # Maintain position in strong trend
            
        # Encourage SELL when trend weakens
        if (bb_pos > high_bb_threshold or trend_r2 < 0.5):
            reward += 30  # Consider selling due to overbought
        elif (rsi_5 > rsi_overbought):
            reward += 30  # Overbought condition, consider selling
        else:
            reward -= 10  # Maintain position
    
    # Adjust reward based on volatility regime
    if vol_ratio > high_volatility_threshold:
        reward -= 20  # Caution in high volatility
    
    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward