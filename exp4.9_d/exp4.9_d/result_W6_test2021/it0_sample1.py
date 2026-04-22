import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]
    price = s[0]  # Current closing price
    sma5 = s[120]
    sma10 = s[121]
    sma20 = s[122]
    rsi10 = s[131]
    rsi14 = s[132]
    
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    regime_volatility = s[144]  # Volatility regime ratio
    trend_r_squared = s[145]  # Trend strength R² of regression
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    # Calculate thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d
    low_vol_threshold = 0.5 * volatility_20d
    
    reward = 0
    
    # Conditions for buying
    if position == 0:  # Not holding
        # Buy signal conditions
        if (price > sma5 and price > sma10 and rsi10 < 30) and (regime_volatility <= 2):
            reward += 50  # Strong buy signal (oversold)
        elif (price > sma5 and price > sma10) and (rsi14 < 40):
            reward += 30  # Potential buy signal

    # Conditions for selling or holding
    elif position == 1:  # Holding
        # Reward for holding during uptrend
        if (price > sma5 and price > sma10 and trend_r_squared > 0.8):
            reward += 10  # Positive reward for holding
        
        # Sell signal conditions
        if (bb_position > 0.8 or rsi14 > 70) and (trend_r_squared < 0.5):
            reward -= 50  # Overbought condition, consider selling
        elif (price < sma20 and trend_r_squared < 0.5):
            reward -= 30  # Trend weakening, consider selling

    # Avoid extremes in rewards
    if reward > 100:
        reward = 100
    elif reward < -100:
        reward = -100

    return reward