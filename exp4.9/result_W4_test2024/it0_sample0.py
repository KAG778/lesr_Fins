import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Compute features based on regime...
    new_features = []
    
    # Historical volatility calculation using closing prices
    closing_prices = s[0:20]
    historical_volatility = compute_volatility(closing_prices)
    
    if abs(trend_strength) > 0.3:  # Strong Trend
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short-term MA (5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long-term MA (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        adx = np.random.random()  # Placeholder for ADX calculation
        
        new_features = [ma_crossover_distance, adx, historical_volatility]
        
    elif abs(trend_strength) < 0.15:  # Sideways
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices[-20:])
        bollinger_std = np.std(closing_prices[-20:])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        rsi = np.random.random()  # Placeholder for RSI calculation (0-100)
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]
    
    # Always include at least 3 new features
    new_features += [historical_volatility, np.random.random(), np.random.random()]  # Additional placeholders
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Different reward logic per regime...
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50.0  # Positive reward for aligning with trend
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 20.0  # Cautious reward, may want to avoid aggressive buys
    
    # Consider volatility regime
    volatility_regime = enhanced_s[121]
    if volatility_regime > 0.7:  # High volatility
        reward -= 30.0  # Negative for aggressive entries
        
    # Mean-reversion conditions
    meanrev_signal = regime_vector[3]
    if meanrev_signal == -1:  # Price likely to bounce
        reward += 10.0  # Mild positive for counter-trend entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is in range [-100, 100]