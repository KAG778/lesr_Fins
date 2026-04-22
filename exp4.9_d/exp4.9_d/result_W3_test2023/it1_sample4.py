import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # Position flag: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day volatility
    volatility_20d = enhanced_state[136]  # 20-day volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    rsi_5d = enhanced_state[128]  # 5-day RSI
    momentum = enhanced_state[134]  # Momentum indicator
    avg_volatility = (volatility_5d + volatility_20d) / 2  # Average volatility for thresholds

    # Initialize reward
    reward = 0
    
    # Define thresholds based on volatility
    high_vol_threshold = 2 * avg_volatility  # High volatility condition
    low_vol_threshold = 0.5 * avg_volatility  # Low volatility condition
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Overbought BB position threshold
    oversold_rsi = 30  # RSI oversold threshold
    overbought_rsi = 70  # RSI overbought threshold

    if position == 0:  # Not holding
        # Reward for clear buy opportunities
        if trend_r_squared > strong_trend_threshold and rsi_5d < oversold_rsi and bb_position < 0.2:
            reward += 50  # Strong buy signal
        elif trend_r_squared > strong_trend_threshold and rsi_5d < 40:
            reward += 30  # Moderate buy signal
        elif volatility_5d < low_vol_threshold:
            reward += 10  # Low volatility, consider buying
        elif volatility_5d > high_vol_threshold:
            reward -= 10  # High volatility, caution

    else:  # Holding
        # Reward for holding during a strong uptrend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 40  # Weak trend, consider selling
        elif bb_position > overbought_threshold or rsi_5d > overbought_rsi:
            reward -= 30  # Overbought condition, consider selling
        elif volatility_5d > high_vol_threshold:
            reward -= 10  # High volatility, caution

    # Normalize the reward to fit within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward