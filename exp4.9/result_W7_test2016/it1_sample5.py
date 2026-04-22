import numpy as np

def compute_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    # Calculate historical volatility
    historical_volatility = compute_historical_volatility(s[0:20])
    
    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        short_ma = np.mean(s[0:10])  # Short moving average (10 days)
        long_ma = np.mean(s[10:20])   # Long moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        # Average True Range (ATR)
        atr = np.mean(s[0:20])  # Placeholder, replace with actual ATR calculation
        trend_consistency = np.sum(np.diff(s[0:20]) > 0) / 19  # Normalized consistency measure
        
        new_features.extend([ma_crossover_distance, atr, trend_consistency])
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        price_bollinger_pct = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        rsi = np.mean(np.diff(s[0:20]) > 0)  # Simple RSI placeholder
        
        new_features.extend([price_bollinger_pct, historical_volatility, rsi])
    
    # Append new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis scenario
    
    # Adjust reward logic based on the regime
    if abs(trend_strength) > 0.3:  # TREND
        if regime_vector[2] > 0:  # Positive momentum
            reward += 50 * trend_strength  # Strong positive for trend alignment
        else:
            reward -= 20 * abs(trend_strength)  # Penalize against trend
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        meanrev_signal = regime_vector[3]
        if meanrev_signal == -1:  # Price at lower Bollinger
            reward += 10  # Positive for counter-trend buy
        elif meanrev_signal == 1:  # Price at upper Bollinger
            reward -= 10  # Negative for counter-trend sell
    
    # Adjust reward for volatility regime
    if regime_vector[1] > 0.7:  # HIGH_VOL
        reward *= 0.5  # Reduce reward magnitude
    
    return np.clip(reward, -100, 100)  # Ensure reward stays within range