import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Relative thresholds based on volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2
    high_volatility_threshold = 2 * avg_volatility  # High volatility condition
    low_volatility_threshold = 0.5 * avg_volatility  # Low volatility condition
    
    # Initialize reward
    reward = 0
    
    if position == 0:  # Not holding stock (Buy logic)
        # Look for strong buy signals
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Decent trend and oversold
            reward += 30  # Moderate buy signal
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 20  # Caution, avoid buying
        else:
            reward -= 5  # Neutral case for indecision

    elif position == 1:  # Holding stock (Sell logic)
        # Look for strong sell signals
        if trend_r2 < 0.5 or bb_pos > 0.8:  # Weak trend or overbought
            reward -= 50  # Strong sell signal
        elif trend_r2 > 0.8:  # Strong trend
            reward += 20  # Reward for holding
        elif bb_pos > 0.5 and trend_r2 < 0.7:  # Moderate trend but nearing overbought
            reward += 10  # Consider taking profit
        elif volatility_5d > high_volatility_threshold:  # High volatility
            reward -= 15  # Caution when holding in high volatility
        else:
            reward += 5  # Mild reward for holding in stable trend

    # Normalize the reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward