import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Initialize new feature list
    new_features = []
    
    # Calculate historical volatility (20-day rolling standard deviation)
    historical_volatility = np.std(s[0:20])  # 20-day standard deviation

    # Feature extraction based on trend
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(s[0:20][-5:])  # Short MA (5 days)
        long_ma = np.mean(s[0:20][-20:])  # Long MA (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        adx = np.std(np.diff(s[0:20]))  # ADX approximation
        trend_consistency = np.sum(np.diff(s[0:20]) > 0) / len(s[0:20])  # Trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.mean(np.diff(s[0:20]) > 0)  # Simplified RSI calculation
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]

    if volatility_regime > 0.7:  # HIGH_VOL regime
        # Add additional volatility features
        new_features.append(historical_volatility)

    if crisis_signal > 0.5:  # CRISIS regime
        # Defensive indicators
        drawdown = np.min(s[0:20]) / np.max(s[0:20]) - 1  # Drawdown calculation
        max_consecutive_losses = np.sum(np.diff(s[0:20]) < 0)  # Count of consecutive losses
        new_features = [drawdown, max_consecutive_losses, trend_strength]

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Different reward logic per regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if momentum_signal > 0:  # Strong upward momentum
            reward += 50.0  # Positive reward for trend-following
        elif momentum_signal < 0:  # Strong downward momentum
            reward -= 20.0  # Penalize counter-trend entries

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if meanrev_signal < 0:  # Price at lower Bollinger
            reward += 20.0  # Positive for mean-reversion opportunities
        elif meanrev_signal > 0:  # Price at upper Bollinger
            reward -= 10.0  # Penalize for chasing breakouts

    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 20.0  # Scale down rewards in high volatility

    # Ensure reward is within range
    return np.clip(reward, -100, 100)  