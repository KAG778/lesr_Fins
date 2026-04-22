import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    position = enhanced_state[150]  # Current position: 1.0 = holding, 0.0 = not holding
    rsi_5 = enhanced_state[128]      # 5-day RSI
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149] # Bollinger Band position [0, 1]
    volatility_5d = enhanced_state[135] # 5-day volatility
    volatility_20d = enhanced_state[136] # 20-day volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2

    reward = 0

    # Define thresholds for trading signals
    rsi_oversold = 30
    rsi_overbought = 70
    strong_trend_threshold = 0.8
    high_volatility_threshold = 2.0 * volatility_20d
    low_volatility_threshold = 0.5 * volatility_20d

    if position == 0:  # Not holding
        # Reward for clear buy opportunities
        if rsi_5 < rsi_oversold and trend_r_squared > strong_trend_threshold:
            reward += 50  # Strong buy signal
        elif rsi_5 < 40 and trend_r_squared > 0.6:
            reward += 20  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:
            reward -= 20  # High volatility, cautious about buying
        elif volatility_5d < low_volatility_threshold:
            reward += 10  # Consider buying in a calm market

    elif position == 1:  # Holding
        # Reward for holding during strong uptrend
        if trend_r_squared > strong_trend_threshold:
            reward += 30  # Good to hold
        # Encourage selling if conditions weaken
        if trend_r_squared < 0.5 or bb_position > 0.8 or rsi_5 > rsi_overbought:
            reward -= 20  # Consider selling due to weakening trend or overbought conditions
        # Caution during high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Be cautious about holding in high volatility

    # Normalize reward to fit within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward