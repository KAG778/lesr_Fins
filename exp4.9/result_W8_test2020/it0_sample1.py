import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and add the regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions base
    
    # Prepare to compute new features based on the regime
    new_features = []
    
    # Ensure we handle edge cases and avoid division by zero
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Moving Average Crossover Distance
        short_ma = np.mean(closing_prices[-5:])  # 5-day MA
        long_ma = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = short_ma - long_ma
        
        # ADX (Average Directional Index) approximation
        # Simplified version; usually requires more data
        directional_movement = np.mean(high_prices[-5] - low_prices[-5]) / np.mean(closing_prices[-5])
        adx = directional_movement * 100  # Scale to percentage
        
        new_features.append(ma_crossover_distance)
        new_features.append(adx)
        new_features.append(np.std(closing_prices))  # Trend consistency via volatility
        
    else:
        # Mean-reversion features
        # Bollinger %B
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        
        # RSI (Relative Strength Index) in neutral zone
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.append(bollinger_percent_b)
        new_features.append(rsi)
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width
    
    # Add new features to the enhanced state
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Determine reward based on regime
    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and positive momentum
            reward += 10  # Positive for trend-following
        elif trend_strength < 0:  # Downtrend
            reward -= 5  # Cautious for downtrend
    elif abs(trend_strength) < 0.15:  # Sideways
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 5  # Mild positive for counter-trend
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 10  # Negative for aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds