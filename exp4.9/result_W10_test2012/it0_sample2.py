import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate additional features based on the regime
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Avoiding division by zero and calculating moving averages and other features
    moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
    moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
    
    # Feature: Moving Average Crossover Distance
    if moving_average_long != 0:
        ma_crossover_distance = (moving_average_short - moving_average_long) / moving_average_long
    else:
        ma_crossover_distance = 0.0
    
    # Feature: Recent volatility (standard deviation of last 5 closing prices)
    recent_volatility = np.std(closing_prices[-5:])
    
    # Feature: Price Range over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices)
    
    if abs(trend_strength) > 0.3:  # Strong Trend
        new_features.append(ma_crossover_distance)
        new_features.append(recent_volatility)
        new_features.append(price_range)
    elif abs(trend_strength) < 0.15:  # Sideways
        # Mean-reversion features
        # Feature: Bollinger %B (assuming a simple calculation)
        bollinger_middle = np.mean(closing_prices[-20:])
        bollinger_std = np.std(closing_prices[-20:])
        upper_bollinger = bollinger_middle + 2 * bollinger_std
        lower_bollinger = bollinger_middle - 2 * bollinger_std
        
        if upper_bollinger != lower_bollinger:
            bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        else:
            bollinger_percent_b = 0.0
            
        new_features.append(bollinger_percent_b)
        new_features.append(recent_volatility)
        new_features.append(price_range)
    
    # If high volatility regime
    if volatility_regime > 0.7:
        # Feature: Average True Range (ATR)
        true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                                 np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                            np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0
        new_features.append(atr)
    
    # Crisis features can be added as required
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on regime
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] == 1:  # Strong upward momentum
            reward = 20.0  # Positive reward
        else:
            reward = 10.0  # Cautious positive reward
    elif trend_strength < -0.3:  # Strong downtrend
        reward = -10.0  # Cautious negative reward
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] == -1:  # Likely to bounce
            reward = 5.0  # Mild positive for mean-reversion opportunity
        else:
            reward = -5.0  # Negative if not a bounce opportunity
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5.0  # Penalty for aggressive entries in high volatility

    return reward