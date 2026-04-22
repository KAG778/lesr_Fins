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
    
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate historical volatility (e.g., using the last 20 days closing prices)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        moving_avg_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        moving_avg_long = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
        ma_crossover_distance = moving_avg_short - moving_avg_long
        
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX, consider more complex calculations
        
        new_features = [ma_crossover_distance, adx, historical_volatility]
        
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        rsi = 100 - (100 / (1 + np.mean(daily_returns[-14:]) / np.std(daily_returns[-14:])))
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]

    # Ensure we have at least 3 features
    new_features = np.array(new_features)
    
    # Handle edge cases
    if len(new_features) < 3:
        new_features = np.pad(new_features, (0, 3 - len(new_features)), 'constant', constant_values=np.nan)
    
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
    if trend_strength > 0.3:
        # Strong uptrend
        if enhanced_s[0] > enhanced_s[20]:  # Current price vs opening price
            reward = 50.0  # Positive for buying in a strong uptrend
        else:
            reward = -10.0  # Cautious if price is not following trend
            
    elif trend_strength < -0.3:
        # Strong downtrend
        if enhanced_s[0] < enhanced_s[20]:  # Current price vs opening price
            reward = 10.0  # Mild positive for shorting in a strong downtrend
        else:
            reward = -20.0  # Cautious if price is not following trend
            
    elif -0.15 <= trend_strength <= 0.15:
        # Sideways market
        if enhanced_s[121] < 0:  # Mean-reversion opportunity
            reward = 20.0  # Positive for counter-trend entries
        else:
            reward = 5.0  # Mild positive for holding
            
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude for high volatility
    
    return reward