import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Compute features based on regime...
    if abs(trend_strength) > 0.3:  # Strong Trend
        # Trend-following features
        short_ma = np.mean(s[0:20])  # short-term moving average (20 days)
        long_ma = np.mean(s[0:60])   # long-term moving average (60 days)
        ma_crossover_distance = short_ma - long_ma
        adx = np.random.rand()  # Placeholder for ADX calculation
        trend_consistency = np.std(s[0:20]) / (np.mean(s[0:20]) + 1e-10)  # Avoid division by zero
        
        new_features = [ma_crossover_distance, adx, trend_consistency]

    elif abs(trend_strength) < 0.15:  # Sideways Market
        # Mean-reversion features
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])
        bollinger_percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid div by zero
        rsi = np.random.rand()  # Placeholder for RSI calculation
        range_width = np.max(s[0:20]) - np.min(s[0:20])
        
        new_features = [bollinger_percent_b, rsi, range_width]

    if volatility_regime > 0.7:  # High Volatility
        # Volatility features
        atr = np.random.rand()  # Placeholder for ATR calculation
        volatility_signal = np.std(s[0:20])  # Historical volatility
        new_features += [atr, volatility_signal]
    
    if crisis_signal > 0.5:  # Crisis
        # Defensive indicators
        drawdown_rate = np.random.rand()  # Placeholder for drawdown calculation
        max_consecutive_losses = np.random.randint(0, 10)  # Placeholder for losses count
        defensive_indicator = np.random.rand()  # Placeholder for defensive calculation
        
        new_features += [drawdown_rate, max_consecutive_losses, defensive_indicator]
    
    # Ensure we return at least 3 new features
    new_features = new_features[:3] if len(new_features) > 3 else new_features + [0] * (3 - len(new_features))

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Different reward logic per regime...
    if trend_strength > 0.3:  # Strong uptrend
        if enhanced_s[125] > 0:  # Assume some momentum feature at index 125
            reward += 20.0  # Strong positive reward for alignment
        else:
            reward -= 5.0  # Cautious negative reward if against momentum
            
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 10.0  # Cautious negative reward
        
    elif abs(trend_strength) < 0.15:  # Sideways market
        if enhanced_s[125] < 0:  # Assume some mean-reversion feature at index 125
            reward += 10.0  # Positive reward for counter-trend entry
            
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5.0  # Negative for aggressive entries

    return reward