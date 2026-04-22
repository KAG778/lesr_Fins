import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # New feature list
    new_features = []

    # Compute features based on the current regime
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Simple Moving Averages (SMA) for trend identification
        short_ma = np.mean(s[0:20][-5:])  # Last 5 days closing price
        long_ma = np.mean(s[0:20][-15:])  # Last 15 days closing price
        ma_crossover = short_ma - long_ma
        
        # Average Directional Index (ADX)
        adx = np.random.uniform(0, 50)  # Placeholder for actual ADX calculation
        
        new_features.append(ma_crossover)
        new_features.append(adx)
        new_features.append(np.mean(s[80:100]))  # Average volume as a feature
        
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        # Bollinger Band %B
        moving_avg = np.mean(s[0:20])
        std_dev = np.std(s[0:20])
        upper_band = moving_avg + 2 * std_dev
        lower_band = moving_avg - 2 * std_dev
        pct_b = (s[0] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
        
        # Relative Strength Index (RSI)
        rsi = np.random.uniform(0, 100)  # Placeholder for actual RSI calculation
        
        new_features.append(pct_b)
        new_features.append(rsi)
        new_features.append(np.max(s[60:80]) - np.min(s[60:80]))  # Range width as a feature
        
    # Append new features to the enhanced state
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
    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0 and enhanced_s[0] > enhanced_s[1]:  # Assuming 0 index is current price, 1 index is previous price
            reward = 10.0  # Positive reward for uptrend
        elif trend_strength < 0 and enhanced_s[0] < enhanced_s[1]:
            reward = 5.0  # Cautious reward for downtrend
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if enhanced_s[0] < enhanced_s[1]:  # Assuming a mean-reversion logic
            reward = 8.0  # Mild positive for counter-trend entries

    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude in high volatility

    return reward