import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Compute features based on regime
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate daily returns to aid in feature calculations
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    
    # Feature calculations
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Simple Moving Average (SMA) Crossover (short-term vs long-term)
        short_sma = np.mean(closing_prices[-5:])  # last 5 days
        long_sma = np.mean(closing_prices[-20:])   # last 20 days
        sma_crossover_distance = short_sma - long_sma
        
        # Average Directional Index (ADX) placeholder (not a full implementation)
        adx = np.std(daily_returns[-14:]) * 100  # Placeholder for ADX-like feature
        new_features.extend([sma_crossover_distance, adx])

    else:
        # Mean-reversion features
        # Bollinger Bands %B
        std_dev = np.std(closing_prices)
        moving_average = np.mean(closing_prices)
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_percent_b = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
        
        # Relative Strength Index (RSI) placeholder for the neutral zone
        rsi = np.mean(daily_returns[-14:]) * 100  # Placeholder for RSI-like feature
        new_features.extend([bollinger_percent_b, rsi])

    # Volatility feature: Average True Range (ATR) placeholder
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
                  )  # Placeholder for ATR calculation
    new_features.append(atr)

    # Combine the features into final enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Different reward logic per regime
    if abs(trend_strength) > 0.3:
        if trend_strength > 0:  # Strong uptrend
            reward += 20  # Positive reward in uptrend
            if enhanced_s[125] > 0:  # Positive momentum
                reward += 30  # Trend is your friend
        else:  # Strong downtrend
            reward -= 10  # Cautious reward in downtrend
            
    elif abs(trend_strength) < 0.15:
        if enhanced_s[126] > 0:  # Mean-reversion opportunity
            reward += 15  # Mild positive for mean-reversion
    
    # Adjust for volatility
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 15  # Aggressive entries are penalized
    
    return reward