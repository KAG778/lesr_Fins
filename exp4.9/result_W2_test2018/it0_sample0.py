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
    
    # Compute features based on regime
    if abs(trend_strength) > 0.3:  # Strong trend
        # Example trend-following features
        short_ma = np.mean(s[0:20])  # Short moving average (last 20 days)
        long_ma = np.mean(s[0:60])  # Long moving average (last 60 days)
        ma_crossover_distance = short_ma - long_ma
        
        # ADX (Average Directional Index) approximation (simple version)
        adx = np.std(s[0:20])  # Placeholder for more complex ADX calculation
        
        new_features = [ma_crossover_distance, adx, np.abs(trend_strength)]
    
    elif abs(trend_strength) < 0.15:  # Sideways
        # Example mean-reversion features
        bollinger_upper = np.mean(s[0:20]) + 2 * np.std(s[0:20])
        bollinger_lower = np.mean(s[0:20]) - 2 * np.std(s[0:20])
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        rsi = np.mean(s[0:20])  # Placeholder for a more comprehensive RSI calculation
        
        new_features = [bollinger_percent_b, rsi, np.std(s[0:20])]
    
    if volatility_regime > 0.7:  # High volatility
        # Example volatility features
        atr = np.mean(np.abs(np.diff(s[0:20])))  # Average True Range approximation
        
        new_features = [atr, np.std(s[0:20]), trend_strength]
    
    if crisis_signal > 0.5:  # In a crisis
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
    
    # Reward based on different regimes
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] == 1:  # Strong upward momentum
            reward += 50.0  # Positive reward
        elif regime_vector[2] == -1:  # Strong downward momentum
            reward -= 20.0  # Cautious reward
    
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 30.0  # Negative reward
    
    elif abs(trend_strength) < 0.15:  # Sideways
        if regime_vector[3] == -1:  # Price at lower Bollinger
            reward += 20.0  # Mild positive for counter-trend entry
        elif regime_vector[3] == 1:  # Price at upper Bollinger
            reward -= 10.0  # Cautious about counter-trend
    
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 20.0  # Negative for aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]