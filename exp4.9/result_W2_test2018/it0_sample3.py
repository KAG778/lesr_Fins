import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate the new features based on the current regime
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Handle edge cases to avoid division by zero
    price_range = np.max(high_prices) - np.min(low_prices)
    if price_range == 0:
        price_range = 1e-6  # Small number to avoid division by zero
    
    # Trend-following features
    if abs(trend_strength) > 0.3:
        # Moving Average Crossover Distance (Example)
        short_ma = np.mean(closing_prices[-5:])  # 5-day MA
        long_ma = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = short_ma - long_ma
        
        # Average True Range (ATR) for volatility adjustment
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # 14-day ATR

        new_features.extend([ma_crossover_distance, atr])
    
    # Mean-reversion features
    else:
        # Bollinger %B
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_b_percent = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.0
        
        # RSI (Relative Strength Index)
        delta = np.diff(closing_prices)
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0.0
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0.0
        rs = gain / loss if loss != 0 else 0.0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.extend([bollinger_b_percent, rsi])
    
    # Volatility-based features
    if volatility_regime > 0.7:
        # Volatility breakout signal
        volatility_breakout = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return percentage
        new_features.append(volatility_breakout)
    
    # Crisis features
    if crisis_signal > 0.5:
        drawdown = np.min(closing_prices) - np.max(closing_prices)  # Simple drawdown calculation
        new_features.append(drawdown)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # Crisis signal handling
    if crisis_signal > 0.5:
        return -50.0  # Strong negative reward during crisis
    
    # Reward logic based on market regime
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] == 1:  # Strong upward momentum
            reward = 20.0  # Positive reward for aligning with trend
        else:
            reward = 5.0  # Cautious reward for being in an uptrend but no momentum
    
    elif trend_strength < -0.3:  # Strong downtrend
        reward = -10.0  # Cautious negative reward in a downtrend
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] == -1:  # Mean reversion opportunity
            reward = 10.0  # Positive reward for counter-trend entry
    
    # High volatility condition
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 15.0  # Negative reward for aggressive entries in high volatility
    
    return reward