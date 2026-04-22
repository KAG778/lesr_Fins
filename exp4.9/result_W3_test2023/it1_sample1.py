import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    closing_prices = s[0:20]
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    if abs(trend_strength) > 0.3:  # TREND regime
        short_ma = np.mean(closing_prices[-5:])  # Short-term MA
        long_ma = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = short_ma - long_ma
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX
        
        new_features = [ma_crossover_distance, adx, historical_volatility]
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid div by zero
        rsi = 100 - (100 / (1 + np.mean(daily_returns[-14:]) / np.std(daily_returns[-14:])))
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]
    
    # Ensure we have at least 3 features
    if len(new_features) < 3:
        new_features += [0] * (3 - len(new_features))
    
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on different regimes
    if trend_strength > 0.3:  # TREND regime
        reward += 50  # Positive reward for aligning with the trend
        if regime_vector[2] > 0:  # Checking momentum
            reward += 20  # Strong upward momentum
        else:
            reward -= 10  # Cautious on downward momentum
            
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 20  # Negative reward for buying in a downtrend
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] == -1:  # Mean-reversion signal
            reward += 20  # Good opportunity to buy low
        else:
            reward -= 10  # Cautious on breakout attempts
    
    # High volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Scale down reward in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]