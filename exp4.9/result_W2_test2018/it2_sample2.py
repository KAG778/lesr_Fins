import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    closing_prices = s[0:20]
    
    # Calculate historical volatility (20-day rolling standard deviation)
    historical_volatility = np.std(closing_prices)
    
    # Feature extraction based on market regime
    if abs(trend_strength) > 0.3:  # Trend regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short moving average (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long moving average (last 20 days)
        ma_crossover_distance = short_ma - long_ma
        adx = np.std(np.diff(closing_prices[-5:]))  # Simplified ADX
        
        new_features = [ma_crossover_distance, adx, historical_volatility]
    
    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + (bollinger_std * 2)
        lower_bollinger = bollinger_mid - (bollinger_std * 2)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        rsi = np.mean(np.diff(closing_prices) > 0)  # Simplified RSI
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]
    
    if volatility_regime > 0.7:  # High volatility regime
        # Additional volatility features
        atr = np.mean(np.abs(np.diff(closing_prices)))  # Average True Range approximation
        new_features.append(atr)
    
    if crisis_signal > 0.5:  # Crisis regime
        # Defensive indicators
        drawdown = np.min(closing_prices) / np.max(closing_prices) - 1  # Drawdown calculation
        max_consecutive_losses = np.sum(np.diff(closing_prices) < 0)  # Count of consecutive losses
        
        new_features = [drawdown, max_consecutive_losses, historical_volatility]
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on different regimes
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 60.0  # Positive reward
        else:  # Weak or negative momentum
            reward -= 20.0  # Cautious entry
    
    elif trend_strength < -0.3:  # Strong downtrend
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 30.0  # Cautious positive reward
        else:  # Weak or positive momentum
            reward -= 50.0  # Strong negative reward
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward += 30.0  # Positive for counter-trend entry
        elif regime_vector[3] > 0:  # Price at upper Bollinger
            reward -= 20.0  # Negative for counter-trend entry
    
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 10.0  # Negative for aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]