import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Calculate new features based on the regime
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    new_features = []

    if abs(trend_strength) > 0.3:  # Strong trend regime
        # Calculate moving averages
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days
        long_ma = np.mean(closing_prices[-20:])  # Last 20 days
        ma_crossover_distance = short_ma - long_ma
        
        # Average Directional Index (ADX) placeholder (real implementation requires more data)
        adx = np.random.rand()  # Placeholder for actual ADX calculation
        
        new_features.append(ma_crossover_distance)
        new_features.append(adx)
        new_features.append(np.mean(np.diff(closing_prices)))  # Trend consistency (average price change)
        
    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_middle + (2 * bollinger_std)
        lower_bollinger = bollinger_middle - (2 * bollinger_std)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        rsi = np.random.rand()  # Placeholder for actual RSI calculation
        range_width = np.max(closing_prices) - np.min(closing_prices)
        
        new_features.append(bollinger_percent_b)
        new_features.append(rsi)
        new_features.append(range_width)
        
    if volatility_regime > 0.7:  # High volatility regime
        # ATR-based feature (placeholder since requires more data)
        atr = np.random.rand()  # Placeholder for actual ATR calculation
        new_features.append(atr)
        
    if crisis_signal > 0.5:  # Crisis regime
        # Defensive features (placeholders)
        drawdown_rate = np.random.rand()  # Placeholder for actual drawdown calculation
        max_consecutive_losses = np.random.randint(1, 10)  # Placeholder for actual losses tracking
        
        new_features.append(drawdown_rate)
        new_features.append(max_consecutive_losses)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic per regime
    if trend_strength > 0.3:  # Uptrend
        reward += 10  # Positive reward for aligning with trend
    elif trend_strength < -0.3:  # Downtrend
        reward -= 5  # Cautious reward, discourage aggressive buying

    if regime_vector[2] == 1:  # Strong upward momentum
        reward += 20
    elif regime_vector[2] == -1:  # Strong downward momentum
        reward -= 15

    if regime_vector[3] == 1:  # Mean-reversion opportunity at upper Bollinger
        reward -= 10  # Negative reward if against mean-reversion

    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5  # Reduce reward for aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Clip reward within range [-100, 100]