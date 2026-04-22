import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # 1.0 = holding stock, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0, 1]
    price_pos = s[146]  # Price position in 20-day range [0, 1]
    
    # Define thresholds for volatility and trading signals
    high_volatility_threshold = 2 * np.mean([volatility_5d, volatility_20d])
    low_volatility_threshold = 0.5 * np.mean([volatility_5d, volatility_20d])
    
    # Initialize reward
    reward = 0

    if position == 0:  # Not holding
        # Reward for clear BUY opportunities
        if trend_r_squared > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.6 and bb_pos < 0.3:  # Moderate uptrend and near oversold
            reward += 30  # Moderate buy signal
        elif volatility_20d < low_volatility_threshold:  # Low volatility
            reward += 10  # Cautious buy signal
        elif bb_pos > 0.8:  # Overbought condition
            reward -= 20  # Avoid buying in overbought condition

    else:  # Holding
        # Reward for HOLD during strong uptrend
        if trend_r_squared > 0.8:
            reward += 30  # Continue holding signal
        elif trend_r_squared < 0.5 and bb_pos > 0.8:  # Weak trend and overbought
            reward -= 50  # Strong sell signal
        elif trend_r_squared < 0.5 or bb_pos > 0.6:  # Weak trend or mildly overbought
            reward -= 20  # Consider selling
        elif volatility_5d > high_volatility_threshold:  # High volatility while holding
            reward -= 30  # Caution in high volatility markets

    # Normalize reward to ensure it remains within [-100, 100]
    return np.clip(reward, -100, 100)