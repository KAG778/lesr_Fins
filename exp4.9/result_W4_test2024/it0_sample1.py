import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Compute new features based on regime
    new_features = []
    
    # Feature computation based on trend
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    
    # Avoid division by zero
    if np.std(closing_prices) != 0:
        momentum = (closing_prices[-1] - closing_prices[-2]) / np.std(closing_prices)  # Momentum relative to historical volatility
    else:
        momentum = 0

    # Trend-following features
    if abs(trend_strength) > 0.3:
        # Example: Calculate moving average crossover distance (short vs long)
        short_ma = np.mean(closing_prices[-5:])  # Short moving average
        long_ma = np.mean(closing_prices[-20:])  # Long moving average
        ma_crossover_distance = short_ma - long_ma
        
        new_features.append(ma_crossover_distance)
        new_features.append(momentum)
        
        # Trend consistency (e.g., standard deviation of returns)
        returns = np.diff(closing_prices) / closing_prices[:-1]
        trend_consistency = np.std(returns)
        new_features.append(trend_consistency)

    # Mean-reversion features
    else:
        # Bollinger %B
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std != 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0
            
        new_features.append(bollinger_percent_b)

        # RSI in neutral zone
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        if avg_loss != 0:
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
        else:
            rsi = 100
            
        new_features.append(rsi)

        # Range width
        range_width = np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])
        new_features.append(range_width)

    # Convert new features list to numpy array
    new_features = np.array(new_features)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Determine reward based on regime
    if trend_strength > 0.3:
        # Strong uptrend
        reward += 10 * trend_strength  # Positive reward aligned with trend
    elif trend_strength < -0.3:
        # Strong downtrend
        reward += -5 * abs(trend_strength)  # Cautious reward
    
    # Sideways regime
    elif abs(trend_strength) < 0.15:
        if enhanced_s[125] < 0.5:  # Assuming last feature is Bollinger %B
            reward += 5  # Mild positive for mean-reversion potential
    
    # High volatility penalty
    if regime_vector[1] > 0.7:
        reward -= 10  # Negative reward for aggressive entries in high volatility
    
    return reward