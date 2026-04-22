import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    
    # Extracting relevant features
    volatility_5d = s[135]   # 5-day volatility
    volatility_20d = s[136]  # 20-day volatility
    trend_r_squared = s[145] # Trend strength (R²)
    momentum = s[134]        # Momentum indicator
    bb_position = s[149]     # Bollinger Band position
    rsi_5d = s[128]          # 5-day RSI

    # Initialize reward
    reward = 0

    # Define dynamic thresholds based on volatility
    high_vol_threshold = 2.0 * volatility_20d  # High volatility threshold
    low_vol_threshold = 0.5 * volatility_20d    # Low volatility threshold
    strong_trend_threshold = 0.8                 # Strong trend R² threshold
    oversold_rsi = 30                            # RSI oversold threshold
    overbought_rsi = 70                          # RSI overbought threshold
    overbought_bb_threshold = 0.8                # Overbought BB position threshold

    # If not holding position (position == 0)
    if position == 0:
        # Strong buy signal
        if rsi_5d < oversold_rsi and trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Moderate buy opportunity
        # Caution against high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 20  # High volatility, exercise caution
        # Consider buying in low volatility
        elif volatility_5d < low_vol_threshold:
            reward += 10  # Buy in calm market

    # If holding position (position == 1)
    else:
        # Reward for holding in strong trend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        # Encourage selling in overbought conditions
        if bb_position > overbought_bb_threshold or rsi_5d > overbought_rsi:
            reward -= 30  # Consider selling to lock in profits
        # Encourage selling when trend weakens
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 40  # Weak trend, consider selling
        # Caution during high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Be cautious about holding in high volatility

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward