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
    
    # Calculate historical volatility (20-day rolling standard deviation)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01
    
    # Feature extraction based on trend strength
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # 5-day MA
        long_ma = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = short_ma - long_ma
        adx = np.std(np.abs(np.diff(closing_prices[-5:])))  # Simplified ADX
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices)  # Trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency, historical_volatility]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.mean(s[1:21])  # Placeholder for a more comprehensive RSI calculation

        new_features = [bollinger_percent_b, rsi, historical_volatility]
    
    if volatility_regime > 0.7:  # HIGH VOLATILITY regime
        # High volatility features
        atr = np.mean(np.abs(np.diff(closing_prices)))  # Average True Range approximation
        new_features.append(atr)

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
    if trend_strength > 0.3:  # TREND UP
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 50.0  # Positive reward for alignment
        else:  # Weak or negative momentum
            reward -= 20.0  # Penalize for counter-trend
    
    elif trend_strength < -0.3:  # TREND DOWN
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 20.0  # Positive for aligning with trend
        else:  # Weak or positive momentum
            reward -= 30.0  # Penalize for counter-trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if regime_vector[3] < 0:  # Price at lower Bollinger band
            reward += 20.0  # Positive for mean-reversion opportunity
        elif regime_vector[3] > 0:  # Price at upper Bollinger band
            reward -= 10.0  # Cautious about counter-trend
    
    if regime_vector[1] > 0.7:  # HIGH VOLATILITY
        reward -= 10.0  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]