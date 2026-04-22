import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def compute_trend_features(closing_prices):
    """Calculate trend-following features."""
    short_ma = np.mean(closing_prices[-5:])  # Last 5 closing prices
    long_ma = np.mean(closing_prices[-20:])  # Last 20 closing prices
    ma_crossover_distance = short_ma - long_ma
    
    # Average True Range (ATR)
    atr = np.mean(np.maximum(
        np.maximum(closing_prices[-5:] - closing_prices[-10:], 0),
        np.maximum(closing_prices[-10:] - closing_prices[-5:], 0)
    ))
    
    return np.array([ma_crossover_distance, atr])

def compute_mean_reversion_features(closing_prices):
    """Calculate mean-reversion features."""
    rolling_mean = np.mean(closing_prices[-20:])
    rolling_std = np.std(closing_prices[-20:])
    
    if rolling_std != 0:
        bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
    else:
        bollinger_percent_b = 0
    
    # RSI Calculation
    gains = np.maximum(np.diff(closing_prices), 0)
    losses = np.maximum(-np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
    
    return np.array([bollinger_percent_b, rsi])

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    closing_prices = s[0:20]  # Last 20 closing prices
    
    if abs(trend_strength) > 0.3:  # Strong trend
        trend_features = compute_trend_features(closing_prices)
        new_features = np.concatenate([trend_features])
    else:  # Sideways
        mean_reversion_features = compute_mean_reversion_features(closing_prices)
        new_features = np.concatenate([mean_reversion_features])
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative for any entries during crisis
    
    # Determine reward based on regime
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0:  # Uptrend
            reward += 50  # Strong positive for aligning with trend
        else:  # Downtrend
            reward -= 20  # Cautious reward, may want to avoid aggressive buys
    
    elif abs(trend_strength) < 0.15:  # Sideways regime
        if enhanced_s[-1] < 0.5:  # Assuming last feature is Bollinger %B
            reward += 10  # Positive for mean-reversion potential
    
    # Adjust rewards based on volatility regime
    if volatility_regime > 0.7:  # High volatility
        reward -= 30  # Negative for aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds