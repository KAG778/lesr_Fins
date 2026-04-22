import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position (1.0 = holding, 0.0 = not holding)
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    
    # Define relative thresholds
    volatility_threshold_buy = 1.5 * volatility_20d  # Caution threshold for buying
    volatility_threshold_sell = 1.5 * volatility_20d  # Caution threshold for selling

    reward = 0

    if position == 0:  # Not holding
        # Strong BUY signal conditions
        if trend_r_squared > 0.8 and bb_position < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r_squared > 0.7 and bb_position < 0.3:  # Good trend with potential
            reward += 30  # Moderate buy signal
        
        # Penalize for high volatility conditions
        if volatility_20d > volatility_threshold_buy:
            reward -= 20  # Be cautious in high volatility

    elif position == 1:  # Holding
        # Reward for holding in strong trends
        if trend_r_squared > 0.8:
            reward += 30  # Strong hold signal
        elif trend_r_squared < 0.5:  # Weakening trend
            reward -= 30  # Consider selling
        
        # Overbought condition
        if bb_position > 0.8:  # Consider selling
            reward -= 40  # Strong signal to sell

        # Penalize for high volatility conditions
        if volatility_20d > volatility_threshold_sell:
            reward -= 15  # Caution on holding during high volatility

    # Normalize to fit in the range [-100, 100]
    return np.clip(reward, -100, 100)