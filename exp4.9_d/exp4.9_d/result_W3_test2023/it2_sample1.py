import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day volatility
    volatility_20d = s[136]  # 20-day volatility
    trend_r_squared = s[145]  # Trend strength (R²)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    rsi_5d = s[128]  # 5-day RSI
    momentum = s[134]  # Momentum indicator
    
    # Initialize reward
    reward = 0
    
    # Define thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d  # High volatility threshold
    low_vol_threshold = 0.5 * volatility_20d  # Low volatility threshold
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Overbought BB position threshold
    oversold_rsi = 30  # RSI oversold threshold
    overbought_rsi = 70  # RSI overbought threshold

    if position == 0:  # Not holding
        # Strong buy signal
        if trend_r_squared > strong_trend_threshold and rsi_5d < oversold_rsi and bb_position < 0.2:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > strong_trend_threshold and rsi_5d < 40:
            reward += 30  # Moderate buy opportunity
        # Caution against high volatility
        elif volatility_5d > high_vol_threshold:
            reward -= 20  # Caution in high volatility, discourage buying
        # Encourage buying in low volatility
        elif volatility_5d < low_vol_threshold:
            reward += 10  # Low volatility, consider buying

    else:  # Holding position
        # Reward for holding in strong trend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold in strong uptrend
        # Encourage selling in overbought conditions or weak trend
        if bb_position > overbought_threshold or rsi_5d > overbought_rsi:
            reward -= 40  # Consider selling to lock in profits
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 30  # Weak trend or negative momentum, consider selling
        # Caution during high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Be cautious about holding in high volatility

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward