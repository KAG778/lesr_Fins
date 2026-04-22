import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state with raw state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions
    
    # Compute new features based on regime
    new_features = []

    # Calculate historical volatility (20-day rolling standard deviation)
    closing_prices = s[0:20]
    if len(closing_prices) > 1:
        historical_volatility = np.std(closing_prices)  # 20-day standard deviation
    else:
        historical_volatility = 0.01  # assume a small value to avoid division by zero
    
    # Feature extraction based on trend
    if abs(trend_strength) > 0.3:
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # 5-day MA
        moving_average_long = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = moving_average_short - moving_average_long
        adx = np.random.uniform(0, 100)  # Placeholder for ADX, replace with actual computation
        
        new_features.extend([ma_crossover_distance, adx, historical_volatility])

    else:
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + (bollinger_std * 2)
        lower_bollinger = bollinger_mid - (bollinger_std * 2)
        bollinger_pct_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        rsi = np.random.uniform(0, 100)  # Placeholder for RSI, replace with actual computation
        
        new_features.extend([bollinger_pct_b, rsi, historical_volatility])

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
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50.0  # Positive reward for uptrend
        if regime_vector[2] > 0:  # Strong momentum
            reward += 30.0  # Add more for positive momentum
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 20.0  # Cautious reward for downtrend
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Positive for counter-trend
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 30.0  # Penalize for aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]