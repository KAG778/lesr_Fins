import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Current position: 1.0 = holding, 0.0 = not holding
    volatility_5d = s[135]  # 5-day historical volatility
    volatility_20d = s[136]  # 20-day historical volatility
    trend_r_squared = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0, 1]
    momentum = s[134]  # Momentum feature
    rsi_14 = s[129]  # 14-day RSI
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d > 0 else 1
    
    # Initialize reward
    reward = 0.0
    
    # Define thresholds
    strong_trend_threshold = 0.8  # Strong trend indicator
    weak_trend_threshold = 0.5  # Weak trend indicator
    overbought_threshold = 0.8  # Overbought condition for Bollinger Bands
    oversold_threshold = 0.2  # Oversold condition for Bollinger Bands
    high_volatility_threshold = 2.0  # High volatility caution threshold

    if position == 0:  # Not holding the stock (BUY signals)
        # Strong buy signal: strong trend and oversold
        if trend_r_squared > strong_trend_threshold and bb_position < oversold_threshold:
            reward += 50  # Strong buy opportunity
        # Moderate buy signal: low volatility and rising momentum
        elif trend_r_squared > weak_trend_threshold and volatility_ratio < high_volatility_threshold and momentum > 0:
            reward += 30  # Moderate buy opportunity
        # Neutral conditions for buying
        else:
            reward -= 10  # Avoid buying in uncertain conditions

    else:  # Holding the stock (SELL signals)
        # Reward for holding during strong trends
        if trend_r_squared > strong_trend_threshold:
            reward += 20  # Encourage holding during strong uptrend
        # Selling conditions based on trend weakening or overbought
        if bb_position > overbought_threshold or trend_r_squared < weak_trend_threshold:
            reward -= 40  # Strong sell signal due to overbought or weak trend
        # Penalize for holding during extreme volatility
        if volatility_ratio > high_volatility_threshold:
            reward -= 20  # Be cautious in high volatility
        
        # Encourage selling if momentum is negative
        if momentum < 0:
            reward -= 30  # Consider selling if momentum is declining

    # Normalize reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward