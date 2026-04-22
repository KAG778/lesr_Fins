import numpy as np

def intrinsic_reward(enhanced_state):
    # Extracting relevant features from the enhanced state
    position = enhanced_state[150]  # 1.0 = holding, 0.0 = not holding
    volatility_5d = enhanced_state[135]  # 5-day volatility
    volatility_20d = enhanced_state[136]  # 20-day volatility
    trend_r_squared = enhanced_state[145]  # Trend strength (R²)
    bb_position = enhanced_state[149]  # Bollinger Band position [0, 1]
    rsi_5d = enhanced_state[128]  # 5-day RSI
    momentum = enhanced_state[134]  # Momentum indicator

    # Initialize reward
    reward = 0
    
    # Calculate volatility ratio
    volatility_ratio = volatility_5d / volatility_20d if volatility_20d != 0 else 0

    # Define thresholds for volatility and trend strength
    high_vol_threshold = 2.0  # High volatility condition
    strong_trend_threshold = 0.8  # Strong trend R² threshold
    oversold_rsi = 30
    overbought_rsi = 70
    overbought_bb_threshold = 0.8  # Overbought Bollinger Band position

    if position == 0:  # Not holding
        # Reward for clear buy signals
        if trend_r_squared > strong_trend_threshold and rsi_5d < oversold_rsi and bb_position < 0.2:
            reward += 50  # Strong buy opportunity
        elif trend_r_squared > strong_trend_threshold and rsi_5d < 40:
            reward += 30  # Moderate buy opportunity
        elif volatility_ratio < 1:  # Low volatility condition
            reward += 10  # Encourage buying in a calm market
        elif volatility_ratio > high_vol_threshold:  # High volatility caution
            reward -= 20  # Be cautious about buying

    else:  # Holding position
        # Reward for holding during a strong uptrend
        if trend_r_squared > strong_trend_threshold and momentum > 0:
            reward += 30  # Good to hold
        # Encourage selling during overbought conditions
        if bb_position > overbought_bb_threshold or rsi_5d > overbought_rsi:
            reward -= 30  # Time to consider selling
        # Encourage selling if trend weakens
        elif trend_r_squared < 0.5:
            reward -= 20  # Weak trend, consider selling

        # Adjust reward based on volatility regime
        if volatility_ratio > high_vol_threshold:  # Extreme market volatility
            reward -= 10  # Be cautious, reduce reward

    # Normalize the reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward