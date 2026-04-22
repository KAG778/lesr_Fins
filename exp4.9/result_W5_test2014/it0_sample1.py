import numpy as np

def compute_trend_features(s):
    # Example calculations for trend-following features
    closing_prices = s[0:20]
    moving_average_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
    moving_average_long = np.mean(closing_prices[-20:])   # Long MA (last 20 days)
    ma_crossover_distance = moving_average_short - moving_average_long
    adx = np.random.random()  # Placeholder for actual ADX calculation
    trend_consistency = np.random.random()  # Placeholder for actual consistency check
    
    return np.array([ma_crossover_distance, adx, trend_consistency])

def compute_mean_reversion_features(s):
    closing_prices = s[0:20]
    bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)  # Upper Bollinger Band
    bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)  # Lower Bollinger Band
    price_bollinger_pct = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
    rsi = np.random.random()  # Placeholder for actual RSI calculation
    range_width = np.max(closing_prices) - np.min(closing_prices)  # Range width
    
    return np.array([price_bollinger_pct, rsi, range_width])

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    if abs(trend_strength) > 0.3:
        # Trend-following features
        new_features = compute_trend_features(s)
    else:
        # Mean-reversion features
        new_features = compute_mean_reversion_features(s)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    momentum_signal = regime_vector[2]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on regime
    if trend_strength > 0.3:
        if momentum_signal > 0:
            reward = 30.0  # Strong uptrend and momentum aligned
        else:
            reward = -10.0  # Strong uptrend but momentum is down (cautious)
    elif trend_strength < -0.3:
        reward = -20.0  # Strong downtrend (cautious)
    else:
        # Sideways market
        if momentum_signal < 0:
            reward = 10.0  # Mean-reversion opportunity
        else:
            reward = 5.0  # Slight positive for holding
        
    # Incorporate volatility regime
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude in high volatility
    
    return reward