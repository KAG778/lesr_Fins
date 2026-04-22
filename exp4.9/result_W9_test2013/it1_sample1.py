import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    # Start with the original state and append the regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate historical volatility (standard deviation of closing prices)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero

    if abs(trend_strength) > 0.3:  # Trend regime
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # 5-day MA
        moving_average_long = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = (moving_average_short - moving_average_long) / (historical_volatility + 1e-5)  # Normalize by volatility
        
        adx = np.mean(np.abs(np.diff(closing_prices)))  # Simplified ADX calculation
        trend_consistency = np.sign(closing_prices[-1] - closing_prices[0])  # Trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency]
        
    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Mean-reversion features
        bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)
        bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower + 1e-5)  # Avoid division by zero
        rsi = 100 - (100 / (1 + np.mean(closing_prices[-5:]) / (np.std(closing_prices[-5:]) + 1e-5)))  # Simplified RSI
        
        new_features = [bollinger_percent_b, rsi, np.max(closing_prices) - np.min(closing_prices)]  # Range width

    # Append the new features to the enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Strong positive reward for following trend
            if regime_vector[2] > 0:  # Momentum aligned
                reward += 20.0  # Extra positive for momentum
        else:  # Downtrend
            reward -= 20.0  # Cautious negative reward for going against the trend

    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 30.0  # Positive for mean-reversion signals
        else:  # Penalize chasing breakouts
            reward -= 10.0  # Mild negative for breakout attempts

    # Adjust for high volatility
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude for aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]