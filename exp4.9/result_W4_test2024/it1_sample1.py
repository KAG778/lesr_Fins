import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def compute_trend_features(s):
    """Compute features for trending markets."""
    short_ma = np.mean(s[0:20])  # Short-term moving average
    long_ma = np.mean(s[0:60])   # Long-term moving average
    ma_crossover_distance = short_ma - long_ma
    
    # Average True Range (ATR) for volatility
    high_prices = s[40:60]
    low_prices = s[60:80]
    close_prices = s[0:20]
    tr = np.maximum(high_prices - low_prices, 
                    np.maximum(np.abs(high_prices - close_prices), 
                               np.abs(low_prices - close_prices)))
    atr = np.mean(tr)
    
    return np.array([ma_crossover_distance, atr])

def compute_mean_reversion_features(s):
    """Compute features for sideways markets."""
    rolling_mean = np.mean(s[0:20])
    rolling_std = np.std(s[0:20])
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bollinger_percent_b = (s[0] - lower_band) / (upper_band - lower_band)  # Current price
    rsi = compute_rsi(s[0:20])  # Compute RSI
    return np.array([bollinger_percent_b, rsi])

def compute_rsi(prices):
    """Compute the Relative Strength Index (RSI)."""
    gains = np.maximum(prices[1:] - prices[:-1], 0)
    losses = np.maximum(prices[:-1] - prices[1:], 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # Base 125 dims
    new_features = []
    
    if abs(trend_strength) > 0.3:  # Strong trend
        new_features = compute_trend_features(s)
    elif abs(trend_strength) < 0.15:  # Sideways
        new_features = compute_mean_reversion_features(s)
    
    # Compute historical volatility for adaptive thresholds
    historical_volatility = compute_volatility(s[0:20])
    new_features = np.concatenate([new_features, [historical_volatility]])
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative reward for entries in crisis

    # Determine reward based on regime
    if trend_strength > 0.3:  # Strong uptrend
        reward += 20.0  # Base positive reward for trend alignment
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 15.0  # Cautious reward for downtrend

    # Sideways regime
    elif abs(trend_strength) < 0.15:
        if enhanced_s[-1] < 0.5:  # Assuming last feature is Bollinger %B
            reward += 10.0  # Positive reward for mean-reversion potential
    
    # High volatility penalty
    if volatility_regime > 0.7:
        reward -= 10.0  # Negative reward for aggressive positions in high volatility
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds