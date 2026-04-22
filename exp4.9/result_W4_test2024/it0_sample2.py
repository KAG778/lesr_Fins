import numpy as np

def compute_trend_features(s):
    # Simple moving averages
    short_ma = np.mean(s[0:20])  # Last 20 closing prices
    long_ma = np.mean(s[0:60])   # Last 60 closing prices
    ma_crossover_distance = short_ma - long_ma
    
    # Average True Range (ATR)
    high_prices = s[40:60]
    low_prices = s[60:80]
    close_prices = s[0:20]
    tr = np.maximum(high_prices - low_prices, 
                    np.maximum(np.abs(high_prices - close_prices), 
                               np.abs(low_prices - close_prices)))
    atr = np.mean(tr)
    
    return np.array([ma_crossover_distance, atr])

def compute_mean_reversion_features(s):
    # Bollinger Bands
    rolling_mean = np.mean(s[0:20])
    rolling_std = np.std(s[0:20])
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bollinger_percent_b = (s[0] - lower_band) / (upper_band - lower_band)  # Current closing price

    # RSI (Relative Strength Index)
    gains = np.maximum(s[0:19] - s[1:20], 0)
    losses = np.maximum(s[1:20] - s[0:19], 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return np.array([bollinger_percent_b, rsi])

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    if abs(trend_strength) > 0.3:  # Strong trend
        new_features = compute_trend_features(s)
    else:  # Sideways
        new_features = compute_mean_reversion_features(s)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        reward += 20  # Base positive reward
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 10  # Cautious reward
    
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5  # Negative for aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds