import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day volatility
    volatility_20d = enhanced_state[136]  # 20-day volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    momentum = enhanced_state[134]  # Momentum indicator
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    rsi_5d = enhanced_state[128]  # 5-day RSI

    # Initialize reward
    reward = 0

    # Define dynamic thresholds based on volatility
    high_vol_threshold = 2.0 * volatility_20d  # High volatility condition
    low_vol_threshold = 0.5 * volatility_20d   # Low volatility condition
    strong_trend_threshold = 0.8                # Strong trend R² threshold
    oversold_rsi = 30                            # RSI oversold threshold
    overbought_rsi = 70                          # RSI overbought threshold
    overbought_bb = 0.8                          # Overbought Bollinger Band position threshold

    # If not holding position (position == 0)
    if position == 0:
        # Reward for strong buy signals
        if rsi_5d < oversold_rsi and trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 50  # Strong buy signal
        # Reward for moderate buy signals
        elif trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Moderate buy signal
        # Caution in high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 20  # Be cautious in high volatility
        elif volatility_5d < low_vol_threshold:
            reward += 10  # Consider buying in low volatility

    # If holding position (position == 1)
    else:
        # Reward for holding in a strong trend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        # Encourage selling in overbought conditions
        if bb_position > overbought_bb or rsi_5d > overbought_rsi:
            reward -= 30  # Time to consider selling
        # Encourage selling if trend weakens
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 40  # Weak trend, consider selling
        # Caution during high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 10  # Be cautious about holding in high volatility

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward