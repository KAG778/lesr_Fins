import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    position = s[150]  # 1 if holding, 0 if not holding

    # Volatility measures
    volatility_5d = s[135]
    volatility_20d = s[136]
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d != 0 else 0
    
    # Trend measures
    trend_r_squared = s[145]
    momentum = s[134]
    
    # Bollinger Band position
    bb_position = s[149]
    
    # RSI metrics
    rsi_5d = s[128]
    
    # Initialize reward
    reward = 0
    
    # Define thresholds
    high_volatility_threshold = 2.0  # High volatility threshold
    strong_trend_threshold = 0.8       # Strong trend threshold
    overbought_threshold = 0.8         # Overbought condition
    oversold_rsi = 30                  # RSI oversold threshold
    overbought_rsi = 70                # RSI overbought threshold

    if position == 0:  # Not holding
        # Buy signal conditions
        if trend_r_squared > strong_trend_threshold and rsi_5d < oversold_rsi and volatility_ratio < high_volatility_threshold:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > 0.6 and rsi_5d < 40:
            reward += 20  # Moderate buy opportunity
        else:
            reward -= 10  # No clear buy opportunity

    else:  # Holding position
        # Hold signal conditions
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold in a strong trend
        elif trend_r_squared < 0.5 or rsi_5d > overbought_rsi or bb_position > overbought_threshold:
            reward -= 30  # Consider selling due to weakening trend or overbought condition
        elif volatility_ratio > high_volatility_threshold:
            reward -= 20  # Caution in high volatility

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward