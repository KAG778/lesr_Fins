import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state with raw state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions
    new_features = []

    # Calculate historical volatility (20-day rolling standard deviation)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero
    
    # Feature extraction based on trend
    if abs(trend_strength) > 0.3:  # Trend regime
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # 5-day MA
        moving_average_long = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = moving_average_short - moving_average_long
        
        adx = np.std(closing_prices[-5:])  # Placeholder for ADX
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices)  # Trend consistency
        
        new_features.extend([ma_crossover_distance, adx, trend_consistency, historical_volatility])

    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + (bollinger_std * 2)
        lower_bollinger = bollinger_mid - (bollinger_std * 2)
        bollinger_pct_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        rsi = np.mean(np.diff(closing_prices[-14:]) > 0)  # Simplified RSI calculation
        
        new_features.extend([bollinger_pct_b, rsi, historical_volatility])

    if volatility_regime > 0.7:  # High volatility features
        atr = np.mean(np.abs(np.diff(closing_prices)))  # Average True Range approximation
        new_features.append(atr)

    if crisis_signal > 0.5:  # In a crisis, include defensive indicators
        drawdown = np.min(closing_prices) / np.max(closing_prices) - 1  # Drawdown calculation
        max_consecutive_losses = np.sum(np.diff(closing_prices) < 0)  # Count of consecutive losses
        new_features.extend([drawdown, max_consecutive_losses, historical_volatility])
    
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
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 50.0  # Positive reward for aligned momentum
        else:  # Weak or negative momentum
            reward -= 20.0  # Penalize counter-trend entries
    
    elif trend_strength < -0.3:  # Strong downtrend
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 10.0  # Mild positive for aligned momentum
        else:  # Weak or positive momentum
            reward -= 30.0  # Strong negative for counter-trend
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward += 20.0  # Positive for mean-reversion entry
        else:  # Price at upper Bollinger
            reward -= 10.0  # Negative for chasing breakouts
    
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 10.0  # Penalize for aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]