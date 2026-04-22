import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    reward = 0
    
    # Position flag
    holding = s[150]  # 1.0 = holding stock, 0.0 = not holding
    
    # Historical Volatility
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    avg_volatility = (volatility_5d + volatility_20d) / 2

    # Trend strength (R²)
    trend_strength = s[145]
    
    # Bollinger Band position
    bb_position = s[149]

    # Price position in the 20-day range
    price_pos = s[146]

    # Reward logic when not holding stock (position = 0)
    if holding == 0:
        # Positive reward for clear BUY opportunities
        if trend_strength > 0.8 and price_pos < 0.3:  # Strong uptrend and near the bottom
            reward += 50  # Strong BUY signal
        elif trend_strength > 0.5 and price_pos < 0.4:  # Moderate uptrend and relatively low position
            reward += 25  # Moderate BUY signal
        elif volatility_5d > 2 * avg_volatility:  # High volatility regime
            reward -= 10  # Caution in extreme market conditions

    # Reward logic when holding stock (position = 1)
    else:
        # Positive reward for holding during uptrend
        if trend_strength > 0.8:
            reward += 30  # Strong HOLD signal
        elif trend_strength < 0.5:
            reward -= 20  # Weak trend, consider selling
        
        # Consider SELL if market is overbought
        if bb_position > 0.8:  # Overbought condition
            reward -= 30  # Encourage to sell
        elif price_pos > 0.7 and trend_strength < 0.6:  # Price is high and trend is weakening
            reward -= 25  # Strong SELL signal

    # Normalize reward to be in the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward