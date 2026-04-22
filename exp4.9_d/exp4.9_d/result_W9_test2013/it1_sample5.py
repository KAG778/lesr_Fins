import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state  # enhanced_state is a 151-dimensional array
    
    # Extract relevant features
    position = s[150]  # Current position: 1 = holding, 0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    price_pos = s[146]  # Price position in the 20-day range [0,1]
    
    # Initialize reward
    reward = 0
    
    # Define thresholds
    high_volatility_threshold = 2.0 * volatility_20d  # High volatility regime
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    overbought_threshold = 0.8  # Bollinger Band overbought threshold
    oversold_threshold = 0.2  # Bollinger Band oversold threshold
    
    if position == 0:  # Not holding
        # Reward logic for buying
        if price_pos < oversold_threshold and trend_r2 > strong_trend_threshold:
            reward += 50  # Strong buy signal
        elif price_pos < 0.5 and trend_r2 > 0.6:
            reward += 30  # Moderate buy signal
        elif trend_r2 < 0.3:  # Weak trend
            reward -= 10  # Penalty for buying in weak conditions
        
        # Caution in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 15  # Caution in high volatility environments

    else:  # Holding
        # Reward logic for holding
        if trend_r2 > strong_trend_threshold:
            reward += 20  # Positive reward for holding in strong trend
        elif trend_r2 < 0.5 or bb_pos > overbought_threshold:
            reward -= 50  # Strong sell signal, encourage selling in weak trend or overbought
        
        # Caution against selling in high volatility
        if volatility_5d > high_volatility_threshold:
            reward -= 10  # Additional caution penalty for holding in high volatility

    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward