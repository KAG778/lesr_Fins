import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Compute features based on the current regime
    if abs(trend_strength) > 0.3:  # Strong Trend
        # Trend-following features
        short_ma = np.mean(s[0:20])  # 20-day moving average
        long_ma = np.mean(s[0:60])   # 60-day moving average
        ma_crossover_distance = short_ma - long_ma  # Distance between MAs
        adx = np.random.uniform(20, 50)  # Placeholder for ADX calculation
        
        new_features.append(ma_crossover_distance)
        new_features.append(adx)
        new_features.append(np.mean(s[80:100]) / np.mean(s[0:20]))  # Volume relative to price
        
    elif abs(trend_strength) < 0.15:  # Sideways
        # Mean-reversion features
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])
        percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger)  # %B
        rsi = np.random.uniform(30, 70)  # Placeholder for RSI calculation
        
        new_features.append(percent_b)
        new_features.append(rsi)
        new_features.append(np.max(s[0:20]) - np.min(s[0:20]))  # Range width
        
    if volatility_regime > 0.7:  # High Volatility
        atr = np.random.uniform(1, 3)  # Placeholder for ATR calculation
        new_features.append(atr)
        new_features.append(np.std(s[0:20]))  # Historical volatility
        
    if crisis_signal > 0.5:  # Crisis
        drawdown_rate = np.random.uniform(0, 30)  # Placeholder for drawdown calculation
        max_consecutive_losses = np.random.randint(1, 5)  # Placeholder
        
        new_features.append(drawdown_rate)
        new_features.append(max_consecutive_losses)
        
    # Convert new_features to a numpy array and concatenate
    new_features = np.array(new_features)
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on the regime
    if trend_strength > 0.3:  # Strong uptrend
        reward += 20.0  # Positive reward for alignment with trend
        if regime_vector[2] > 0:  # Momentum aligned
            reward += 30.0  # Extra reward for strong upward momentum
    
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 10.0  # Cautious reward, reduce entry signals
    
    elif abs(trend_strength) <= 0.15:  # Sideways
        if regime_vector[3] < 0:  # Mean-reversion opportunity
            reward += 15.0  # Mild positive for counter-trend entries
    
    if regime_vector[1] > 0.7:  # High Volatility
        reward -= 20.0  # Negative for aggressive entries
    
    return reward