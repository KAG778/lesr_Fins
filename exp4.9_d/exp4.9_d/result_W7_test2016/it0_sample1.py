import numpy as np

def intrinsic_reward(enhanced_state):
    # Extract relevant features from the enhanced state
    s = enhanced_state
    
    # Current position: 1.0 if holding, 0.0 if not
    position = s[150]
    
    # Simple Moving Averages
    sma5 = s[120]
    sma20 = s[122]
    
    # Price and trend indicators
    current_price = s[0]  # latest close price
    price_sma20_deviation = s[124]  # Price/SMA20 deviation
    r_squared = s[145]  # Trend strength (R² of regression)
    
    # Volatility metrics
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    volatility_ratio = s[144]  # Volatility regime ratio
    
    # Bollinger Band position
    bb_pos = s[149]  # Bollinger Band position [0,1]
    
    # Initialize reward
    reward = 0.0
    
    # Calculate relative thresholds for volatility
    high_volatility_threshold = 2 * volatility_20d
    low_volatility_threshold = 0.5 * volatility_20d
    
    # Reward Logic
    if position == 0:  # Not holding
        # Buying conditions
        if r_squared > 0.8 and bb_pos < 0.2:  # Strong trend and oversold
            reward += 50
        elif price_sma20_deviation < -0.02:  # Price significantly below SMA20
            reward += 30
        elif volatility_ratio < low_volatility_threshold:  # Low volatility
            reward += 20
    else:  # Holding position
        # Selling conditions
        if r_squared < 0.5 and bb_pos > 0.8:  # Weak trend and overbought
            reward -= 50
        elif price_sma20_deviation > 0.02:  # Price significantly above SMA20
            reward -= 30
        elif volatility_ratio > high_volatility_threshold:  # High volatility
            reward -= 20
        else:  # Encourage holding during strong trends
            if r_squared > 0.8:
                reward += 10  # Small positive for holding in a strong trend
            else:
                reward -= 10  # Small penalty for holding in uncertain conditions

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward