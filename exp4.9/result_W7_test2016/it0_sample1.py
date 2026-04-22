import numpy as np

def calculate_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Compute features based on regime...
    new_features = []
    
    # Calculate historical volatility from closing prices
    historical_volatility = calculate_historical_volatility(s[0:20])
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        short_ma = np.mean(s[0:10])  # Short moving average (10 days)
        long_ma = np.mean(s[10:20])   # Long moving average (20 days)
        ma_crossover_distance = short_ma - long_ma  # Distance between MAs
        adx = np.random.random()  # Placeholder for ADX calculation
        
        new_features.extend([ma_crossover_distance, adx, historical_volatility])
        
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])  # Middle Bollinger band
        bollinger_std = np.std(s[0:20])  # Standard deviation
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        new_features.extend([bollinger_percent_b, historical_volatility])
    
    # Handle edge cases for new features
    new_features = np.nan_to_num(np.array(new_features), nan=0.0)  # Replace NaNs with 0
    
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
        reward += 10 * (1 + trend_strength)  # Positive reward for alignment
        
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 5 * abs(trend_strength)  # Cautious approach
        
    elif abs(trend_strength) < 0.15:  # Sideways market
        meanrev_signal = regime_vector[3]
        if meanrev_signal == -1:  # Price at lower Bollinger
            reward += 5  # Mild positive for counter-trend buy
        elif meanrev_signal == 1:  # Price at upper Bollinger
            reward -= 5  # Negative for counter-trend sell
        
    # High volatility reduces reward magnitude
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude
        
    return np.clip(reward, -100, 100)  # Ensure reward stays within range