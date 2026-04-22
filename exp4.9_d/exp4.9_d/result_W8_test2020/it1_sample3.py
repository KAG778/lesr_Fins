import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    position = enhanced_state[150]  # Current position (1 = holding, 0 = not holding)
    volatility_5d = enhanced_state[135]  # 5-day historical volatility
    volatility_20d = enhanced_state[136]  # 20-day historical volatility
    trend_r2 = enhanced_state[145]  # Trend strength (R² of regression)
    bb_pos = enhanced_state[149]  # Bollinger Band position [0, 1]
    
    # Calculate relative thresholds based on volatility
    high_volatility_threshold = 2 * volatility_20d  # High volatility condition
    low_volatility_threshold = 0.5 * volatility_20d  # Low volatility condition

    # Initialize reward
    reward = 0.0

    if position == 0:  # Not holding stock (BUY opportunities)
        # Look for strong buy conditions
        if trend_r2 > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50  # Strong buy signal
        elif trend_r2 > 0.6 and bb_pos < 0.3:  # Good trend and slightly oversold
            reward += 30  # Moderate buy opportunity
        elif volatility_5d > high_volatility_threshold:  # High volatility caution
            reward -= 20  # Penalize for buying in high volatility
        else:
            reward -= 5  # Neutral case, discourage random buys

    elif position == 1:  # Holding stock (HOLD or SELL opportunities)
        # Look for hold or sell signals
        if trend_r2 > 0.8:  # Strong trend, encourage holding
            reward += 20  # Reward for holding
        elif bb_pos > 0.8:  # Overbought condition, consider selling
            reward -= 30  # Penalize for not selling
        elif trend_r2 < 0.5:  # Weak trend, consider selling
            reward -= 20  # Encourage selling in weak trends
        elif volatility_5d > high_volatility_threshold:  # High volatility caution
            reward -= 10  # Caution when holding in high volatility
        else:
            reward += 5  # Small reward for maintaining position in stable conditions

    # Normalize reward to be within [-100, 100]
    return np.clip(reward, -100, 100)