import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    position = s[150]  # Current position (1 = holding, 0 = not holding)
    
    # Extract relevant features
    volatility = np.mean(s[135:137])  # Mean of 5-day and 20-day historical volatility
    trend_strength = s[145]  # R² of the trend
    bb_position = s[149]  # Bollinger Band position
    price_position = s[146]  # Price position in the 20-day range [0, 1]
    
    # Define thresholds
    high_volatility_threshold = 2 * volatility
    strong_trend_threshold = 0.8
    overbought_threshold = 0.8
    oversold_threshold = 0.2
    
    reward = 0
    
    if position == 0:  # Not holding stock
        # Encourage buying on strong uptrend or oversold
        if trend_strength > strong_trend_threshold and price_position < oversold_threshold:
            reward += 20  # Clear BUY opportunity
        elif trend_strength > strong_trend_threshold and price_position < 0.5:
            reward += 10  # Moderate BUY opportunity
        elif price_position > overbought_threshold and trend_strength < strong_trend_threshold:
            reward -= 10  # Avoid buying in overbought conditions
        
    elif position == 1:  # Holding stock
        # Encourage holding in uptrend
        if trend_strength > strong_trend_threshold:
            reward += 15  # Continue holding
        
        # Encourage selling when in overbought condition or trend weakens
        if bb_position > overbought_threshold:
            reward -= 20  # Strong signal to sell
        elif trend_strength < strong_trend_threshold:
            reward -= 15  # Weakening trend, consider selling
    
    # Penalize extreme market conditions to encourage caution
    if volatility > high_volatility_threshold:
        reward -= 10  # Caution in high volatility
    
    # Ensure reward is within the bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward