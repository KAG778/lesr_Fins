import numpy as np

def calculate_historical_volatility(prices):
    """ Calculate the historical volatility given closing prices. """
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # New feature list
    new_features = []
    
    closing_prices = s[0:20]
    
    # Calculate historical volatility from closing prices
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # Distance between MAs
        adx = np.random.rand()  # Placeholder for ADX calculation
        
        new_features.extend([ma_crossover_distance, adx, historical_volatility])
    else:
        # Mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std
        bollinger_lower = bollinger_middle - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.random.rand()  # Placeholder for RSI calculation
        
        new_features.extend([bollinger_percent_b, rsi, historical_volatility])
    
    # Append the new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Different reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        if enhanced_s[-3] > 0:  # Example of checking a trend-following signal
            reward += 20  # Positive reward for aligned trend
        else:
            reward -= 10  # Cautious if not aligned
    
    elif trend_strength < -0.3:  # Strong downtrend
        reward += 5  # Cautious reward for downtrend
    
    elif abs(trend_strength) < 0.15:  # Sideways
        if enhanced_s[-4] < 0:  # Example of a mean-reversion signal
            reward += 10  # Mild positive for mean-reversion opportunity
    
    # Adjust for high volatility
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 15  # Negative for aggressive entries
        
    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range