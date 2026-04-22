import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Compute features based on regime...
    new_features = []
    
    # Feature calculations
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate 20-day moving average
    moving_average = np.mean(closing_prices)
    
    # Calculate price range
    price_range = np.max(high_prices) - np.min(low_prices)
    
    # Calculate the percentage of price above moving average
    price_above_ma = (closing_prices[-1] - moving_average) / moving_average if moving_average != 0 else 0
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Average True Range (ATR) for volatility
        true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                                 np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                            np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
        new_features.extend([moving_average, price_above_ma, atr])
        
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        # Bollinger Bands %B
        std_dev = np.std(closing_prices)
        bollinger_upper = moving_average + (2 * std_dev)
        bollinger_lower = moving_average - (2 * std_dev)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) != 0 else 0
        rsi = (np.sum(closing_prices[-14:] > np.mean(closing_prices[-14:])) / 14) * 100  # Simplified RSI
        new_features.extend([price_range, bollinger_percent_b, rsi])
    
    # Ensure to return concatenated state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Different reward logic per regime...
    if trend_strength > 0.3:
        # Strong uptrend
        reward += 10.0 if regime_vector[2] > 0 else -5.0  # Momentum aligns positively
    elif trend_strength < -0.3:
        # Strong downtrend
        reward += -5.0  # Cautious reward
    else:
        # Sideways market
        if regime_vector[3] < 0:
            reward += 5.0  # Mild positive for mean-reversion opportunity
        else:
            reward += -2.0  # Neutral to negative for lack of opportunity
    
    # High volatility adjustment
    if regime_vector[1] > 0.7:
        reward *= 0.5  # Reduce reward magnitude in high volatility
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds