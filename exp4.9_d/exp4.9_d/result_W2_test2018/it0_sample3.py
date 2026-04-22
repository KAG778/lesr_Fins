import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features from enhanced_state
    position = s[150]
    
    # Price and trend indicators
    price = s[0]  # last closing price
    sma5 = s[120]  # 5-day SMA
    sma10 = s[121]  # 10-day SMA
    r_squared = s[145]  # Trend strength (R² of regression)
    bb_pos = s[149]  # Bollinger Band position
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    
    # Calculate thresholds for decision making
    high_volatility_threshold = 2 * volatility_20d
    strong_trend_threshold = 0.8  # R² threshold for strong trend
    overbought_threshold = 0.8  # Bollinger Band position threshold

    reward = 0  # Initialize reward

    # Reward logic based on position
    if position == 0:  # Not holding
        # Identify BUY opportunities
        if price > sma5 and r_squared > strong_trend_threshold:  # Uptrend signal
            reward += 50  # Strong BUY signal
        elif price < sma5 and bb_pos < 0.2:  # Oversold condition
            reward += 30  # Moderate BUY signal
        else:
            reward -= 10  # Neutral or weak signal leads to a small penalty
            
    elif position == 1:  # Holding
        # Encourage HOLD during uptrend
        if price > sma10 and r_squared > strong_trend_threshold:
            reward += 40  # Strong HOLD signal
        elif price < sma10 and bb_pos > overbought_threshold:  # Overbought condition
            reward += 20  # Consider SELL signal
        elif r_squared < strong_trend_threshold:  # Weak trend
            reward -= 20  # Consider SELL to avoid losses
        else:
            reward += 10  # Small reward for maintaining position in a strong trend

    # Incorporate volatility considerations
    if volatility_5d > high_volatility_threshold:
        reward -= 20  # Caution in high volatility markets

    # Clamp reward to [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward