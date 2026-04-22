import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    s = enhanced_state
    
    # Extract relevant features from the enhanced state
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    trend_strength = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    vol_5d = s[135]  # 5-day historical volatility
    vol_20d = s[136]  # 20-day historical volatility
    avg_volatility = (vol_5d + vol_20d) / 2  # Average volatility
    vol_ratio = s[144]  # Volatility regime ratio

    # Initialize reward
    reward = 0.0
    
    # Define thresholds for trading signals
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    strong_trend_threshold = 0.8  # Strong trend threshold
    oversold_threshold = 0.2  # Bollinger Band position for oversold
    overbought_threshold = 0.8  # Bollinger Band position for overbought

    if position == 0:  # Not holding stock (BUY phase)
        if trend_strength > strong_trend_threshold and bb_pos < oversold_threshold:
            reward += 50  # Strong buy signal
        elif trend_strength > strong_trend_threshold:
            reward += 15  # Moderate buy signal if not oversold
        elif vol_ratio > high_volatility_threshold:
            reward -= 20  # Caution in high volatility
        elif trend_strength < 0.5:
            reward -= 10  # Weak trend caution
    
    elif position == 1:  # Holding stock (SELL/HOLD phase)
        if trend_strength > strong_trend_threshold:
            reward += 20  # Encourage holding in strong trends
        elif bb_pos > overbought_threshold:
            reward -= 30  # Consider selling due to overbought
        elif trend_strength < 0.5:
            reward -= 25  # Weak trend, encourage selling
        elif vol_ratio > high_volatility_threshold:
            reward -= 15  # Caution advised in extreme volatility

    # Normalize the reward to fit the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward