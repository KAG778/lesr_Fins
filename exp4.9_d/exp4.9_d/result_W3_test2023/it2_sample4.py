import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    rsi_5d = enhanced_state[128]  # 5-day RSI
    momentum = enhanced_state[134]  # Momentum indicator

    # Initialize reward
    reward = 0

    # Define thresholds for trading signals
    high_volatility_threshold = 2.0 * volatility_20d
    low_volatility_threshold = 0.5 * volatility_20d
    strong_trend_threshold = 0.8
    oversold_rsi = 30
    overbought_rsi = 70
    overbought_bb_threshold = 0.8

    if position == 0:  # Not holding (potential buy)
        # Strong buy signal
        if trend_r_squared > strong_trend_threshold and rsi_5d < oversold_rsi and bb_position < 0.2:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal
        elif trend_r_squared > strong_trend_threshold and rsi_5d < 40:
            reward += 30  # Moderate buy opportunity
        # Caution in high volatility
        elif volatility_5d > high_volatility_threshold:
            reward -= 20  # High volatility, exercise caution
        # Consider buying in low volatility
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Low volatility, consider buying

    else:  # Holding position
        # Reward for holding during a strong uptrend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        # Encourage selling if conditions weaken
        if trend_r_squared < 0.5 or momentum < 0 or bb_position > overbought_bb_threshold or rsi_5d > overbought_rsi:
            reward -= 30  # Consider selling due to weakening trend or overbought condition
        # Caution during high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Be cautious about holding in high volatility

    # Normalize the reward to fit within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward