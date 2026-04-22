import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extract relevant features
    position = s[150]  # Position flag (1.0 = holding, 0.0 = not holding)
    
    # Volatility metrics
    historical_volatility_5d = s[135]  # 5-day historical volatility
    historical_volatility_20d = s[136]  # 20-day historical volatility
    avg_historical_volatility = (historical_volatility_5d + historical_volatility_20d) / 2
    
    # Regime metrics
    vol_regime = s[144]  # Volatility regime ratio
    trend_r2 = s[145]  # Trend strength (R² of regression)
    bb_position = s[149]  # Bollinger Band position [0,1]
    
    reward = 0
    
    # Reward logic
    if position == 0:
        # Not holding position (BUY opportunities)
        if trend_r2 > 0.8 and bb_position < 0.2:  # Strong uptrend and oversold
            reward += 50  # Strong BUY signal
        elif trend_r2 > 0.6 and s[134] < 0:  # Momentum is negative but trend is decent
            reward += 10  # Mild BUY signal
    else:
        # Holding position (HOLD or SELL opportunities)
        if trend_r2 > 0.8:  # Strong trend
            reward += 10  # Reward for holding
        elif trend_r2 < 0.5 or bb_position > 0.8:  # Weak trend or overbought
            reward -= 50  # Strong SELL signal
        else:
            reward += 5  # Mild reward for holding in a somewhat stable trend
    
    # Caution in high volatility environments
    if vol_regime > 2:
        reward -= 20  # Penalize in extreme volatility
    
    # Normalize the reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward