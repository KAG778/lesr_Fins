import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    # Calculate the last 20 closing prices for feature extraction
    closing_prices = s[:20]
    
    # Calculate historical volatility for adaptive thresholds
    historical_volatility = np.std(np.log(closing_prices[1:] / closing_prices[:-1])) if len(closing_prices) > 1 else 0
    
    # Thresholds for feature conditioning based on volatility
    trend_threshold_up = 0.3 * historical_volatility
    trend_threshold_down = -0.3 * historical_volatility
    sideways_threshold = 0.15 * historical_volatility
    
    if abs(trend_strength) > trend_threshold_up:
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short moving average (5 days)
        long_ma = np.mean(closing_prices[-20:])   # Long moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / 19  # Normalized trend consistency
        
        new_features.extend([ma_crossover_distance, trend_consistency])
        
    elif abs(trend_strength) < sideways_threshold:
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        
        new_features.extend([bollinger_percent_b])
        
    # Handle edge cases: Replace NaNs with 0
    new_features = np.nan_to_num(np.array(new_features), nan=0.0)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # Critical: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative for entries in crisis
    
    # Reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50 * trend_strength  # Strong positive reward for momentum-aligned entries
        
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 20 * abs(trend_strength)  # Penalize aggressive buying
        
    elif abs(trend_strength) < 0.15:  # Sideways market
        meanrev_signal = regime_vector[3]
        if meanrev_signal < 0:  # Suggesting a buying opportunity
            reward += 10  # Positive for buying at lower Bollinger
        elif meanrev_signal > 0:  # Suggesting a selling opportunity
            reward -= 10  # Negative for selling at upper Bollinger

    # High volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Scale down all rewards in high volatility scenarios
    
    return np.clip(reward, -100, 100)  # Ensure reward stays within range