import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0.0
    
    # Extracting relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]   # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]        # Trend strength (R² of regression)
    bb_pos = s[149]          # Bollinger Band position [0,1]
    
    # Define thresholds
    high_volatility_ratio = 2.0  # High volatility indicates caution
    strong_trend_r2 = 0.8         # Clear trend
    overbought_bb_threshold = 0.8  # Overbought condition
    
    # Reward calculations based on position
    if position == 0:  # Not holding
        # Look for clear BUY opportunities
        if trend_r2 > strong_trend_r2:  # Strong trend
            reward += 50  # Positive reward for trend-following
        if bb_pos < 0.2:  # Oversold condition
            reward += 30  # Additional positive reward for buying at low prices
        # Penalize for buying in high volatility regime
        if volatility_5d / volatility_20d > high_volatility_ratio:
            reward -= 20  # Caution in extreme volatility
    else:  # Holding
        # Reward for holding during an uptrend
        if trend_r2 > strong_trend_r2:  # Strong trend
            reward += 20  # Positive reward for holding
        # Consider selling if overbought or trend weakens
        if bb_pos > overbought_bb_threshold:  # Overbought condition
            reward -= 30  # Penalize for holding in overbought condition
        if volatility_5d / volatility_20d > high_volatility_ratio:  # High volatility
            reward -= 20  # Caution in extreme volatility
        # Consider selling if trend weakens
        if trend_r2 < 0.5:  # Weak trend
            reward -= 40  # Strong incentive to sell if trend weakens

    # Normalize reward to [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward