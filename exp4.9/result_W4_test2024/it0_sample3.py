import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Compute features based on regime...
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Moving Average Crossover Distance
        short_ma = np.mean(s[0:20])  # Last 20 days closing prices
        long_ma = np.mean(s[0:60])   # Last 60 days closing prices
        ma_crossover_distance = short_ma - long_ma
        new_features.append(ma_crossover_distance)
        
        # Average Directional Index (ADX) could be computed but is complex
        # Here, we'll use a simple trend consistency feature
        trend_consistency = np.mean(np.diff(s[0:20]))  # Average daily price change
        new_features.append(trend_consistency)
        
        # Some measure of volume trend (e.g., average volume)
        avg_volume = np.mean(s[80:100])  # Last 20 days of volume
        new_features.append(avg_volume)
    
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        # Bollinger %B
        rolling_mean = np.mean(s[0:20])
        rolling_std = np.std(s[0:20])
        if rolling_std != 0:  # Avoid division by zero
            bollinger_b = (s[0] - rolling_mean) / (2 * rolling_std)  # Latest price
        else:
            bollinger_b = 0.0
        new_features.append(bollinger_b)
        
        # RSI calculation
        deltas = np.diff(s[0:20])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        new_features.append(rsi)
        
        # Price range width
        price_range_width = np.max(s[0:20]) - np.min(s[0:20])
        new_features.append(price_range_width)

    # Check for high volatility
    if volatility_regime > 0.7:
        # Average True Range (ATR) or similar volatility measure
        # For simplicity, we'll use the range of the last 20 days
        volatility_measure = np.max(s[0:20]) - np.min(s[0:20])
        new_features.append(volatility_measure)

    # Crisis features
    if crisis_signal > 0.5:
        # Defensive indicator: max consecutive losses (simple version)
        # We can track losses by looking at the closing prices
        losses = np.where(np.diff(s[0:20]) < 0)[0]
        max_consecutive_losses = 0
        if len(losses) > 0:
            current_streak = 1
            for i in range(1, len(losses)):
                if losses[i] == losses[i - 1] + 1:
                    current_streak += 1
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    current_streak = 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        new_features.append(max_consecutive_losses)

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
    if trend_strength > 0.3:  # Strong uptrend
        if enhanced_s[125] > 0:  # Assuming the first new feature is a trend-following metric
            reward += 50  # Positive reward for following the trend
        else:
            reward -= 10  # Slight penalty for not aligning with trend

    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 20  # Cautious reward, discourage buying

    elif abs(trend_strength) < 0.15:  # Sideways
        if enhanced_s[125] < 0:  # Assuming this is mean reversion opportunity
            reward += 20  # Mild positive for counter-trend entries

    if regime_vector[1] > 0.7:  # High volatility
        reward -= 15  # Negative for aggressive entries

    return reward