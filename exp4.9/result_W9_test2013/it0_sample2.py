import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate the moving averages (MA) and their distances
    closing_prices = s[0:20]
    moving_average_short = np.mean(closing_prices[-5:])  # Short MA (5 days)
    moving_average_long = np.mean(closing_prices[-20:])  # Long MA (20 days)
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # MA Crossover distance
        ma_crossover_distance = moving_average_short - moving_average_long
        new_features.append(ma_crossover_distance)
        
        # Average Directional Index (ADX) can be approximated
        adx = np.random.uniform(20, 50)  # Placeholder for ADX calculation
        new_features.append(adx)
        
        # Trend consistency (simple measure)
        trend_consistency = np.sign(closing_prices[-1] - closing_prices[0])
        new_features.append(trend_consistency)
        
    else:
        # Mean-reversion features
        # Bollinger %B
        std_dev = np.std(closing_prices)
        bollinger_upper = moving_average_short + (2 * std_dev)
        bollinger_lower = moving_average_short - (2 * std_dev)
        
        if (bollinger_upper - bollinger_lower) > 0:  # Avoid division by zero
            bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        else:
            bollinger_percent_b = 0.0
        
        new_features.append(bollinger_percent_b)
        
        # RSI (Relative Strength Index)
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        
        avg_gain = np.mean(gains[-14:])  # Using the last 14 days for RSI
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        new_features.append(rsi)

    # Ensure new features are NumPy array
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
    
    # Different reward logic per regime...
    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0 and enhanced_s[0] < enhanced_s[1]:  # Price is trending up
            reward += 20.0  # Positive reward for uptrend
        elif trend_strength < 0 and enhanced_s[0] > enhanced_s[1]:  # Price is trending down
            reward -= 10.0  # Cautious reward for downtrend
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if enhanced_s[121] < 0.5:  # Assuming some mean-reversion opportunity
            reward += 10.0  # Mild positive for counter-trend entries

    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5.0  # Negative for aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]