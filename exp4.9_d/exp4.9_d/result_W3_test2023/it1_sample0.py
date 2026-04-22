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

    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d
    low_volatility_threshold = 0.5 * volatility_20d
    strong_trend_threshold = 0.8
    overbought_threshold = 0.8
    oversold_threshold = 30

    # If not holding position (position == 0)
    if position == 0:
        # Strong buy signal
        if rsi_5d < oversold_threshold and trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 50  # Strong buy signal
        # Moderate buy signal
        elif trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Moderate buy signal
        # Caution against high volatility
        elif volatility_5d > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
        # Consider buying in low volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Buy in calm market

    # If holding position (position == 1)
    else:
        # Reward for holding in strong trend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        # Encourage selling in overbought conditions
        elif bb_position > overbought_threshold or rsi_5d > 70:
            reward -= 30  # Consider selling to lock in profits
        # Encourage selling when trend weakens
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 40  # Weak trend, consider selling
        # Caution in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Caution in high volatility

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward