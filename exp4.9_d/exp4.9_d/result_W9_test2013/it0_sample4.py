import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array

    # Extract relevant features
    position = s[150]
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in 20-day range [0,1]

    # Define reward variables
    reward = 0

    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility regime
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold

    # Reward logic based on the position
    if position == 0:  # Not holding
        # Positive reward for clear BUY opportunities
        if price_pos < oversold_threshold and trend_r_squared > strong_trend_threshold:
            reward += 50  # Strong buy signal
        elif price_pos < 0.5 and trend_r_squared > strong_trend_threshold:
            reward += 20  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:
            reward -= 10  # Caution in high volatility
    else:  # Holding
        # Reward for HOLDing during an uptrend
        if trend_r_squared > strong_trend_threshold and price_pos > 0.5:
            reward += 20  # Continue holding in a strong trend
        elif trend_r_squared < 0.4 or (bb_position > overbought_threshold):
            reward += 10  # Signal to consider selling
            reward -= 20  # Penalize if the price is too high (overbought)
        elif price_pos < oversold_threshold:
            reward -= 15  # Potential signal to sell if reversing trend

    # Normalize reward to fit within the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward