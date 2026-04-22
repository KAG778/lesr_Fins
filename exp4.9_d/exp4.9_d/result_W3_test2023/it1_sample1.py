import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # R² of trend regression
    bb_position = s[149]  # Bollinger Band position [0, 1]
    rsi_5d = s[128]  # 5-day RSI
    momentum = s[134]  # Momentum indicator

    # Initialize reward
    reward = 0.0
    
    # Calculate dynamic thresholds based on volatility
    high_vol_threshold = 2 * volatility_20d
    low_vol_threshold = 0.5 * volatility_20d
    
    # Logic for not holding a position
    if position == 0:  
        # Strong Buy signal
        if trend_r_squared > 0.8 and rsi_5d < 30 and bb_position < 0.2:
            reward += 50  # Strong buy opportunity
        # Moderate Buy signal
        elif trend_r_squared > 0.6 and rsi_5d < 40:
            reward += 30  # Moderate buy opportunity
        # Caution in high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 30  # High volatility, exercise caution
        elif volatility_5d < low_vol_threshold:
            reward += 10  # Low volatility, opportunity for a buy
        
    # Logic for holding a position
    elif position == 1:  
        # Reward for holding in strong uptrend
        if trend_r_squared > 0.8 and momentum > 0:
            reward += 30  # Continue holding in a strong trend
        # Consider selling in overbought conditions or weak trend
        if bb_position > 0.8 or rsi_5d > 70:
            reward -= 40  # Overbought condition, consider selling
        elif trend_r_squared < 0.5 or momentum < 0:
            reward -= 30  # Weak trend, encourage selling
        # Caution in high volatility
        if volatility_5d > high_vol_threshold:
            reward -= 10  # High volatility, exercise caution

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward