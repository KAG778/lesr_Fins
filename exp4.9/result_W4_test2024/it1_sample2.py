import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def compute_trend_features(s):
    """Compute trend-following features."""
    closing_prices = s[0:20]
    short_ma = np.mean(closing_prices[-5:])  # Short moving average
    long_ma = np.mean(closing_prices[-20:])  # Long moving average
    ma_crossover_distance = short_ma - long_ma
    
    returns = np.diff(closing_prices) / closing_prices[:-1]
    trend_consistency = np.std(returns)
    
    return np.array([ma_crossover_distance, trend_consistency])  # Keep it simple with two features

def compute_mean_reversion_features(s):
    """Compute mean-reversion features."""
    closing_prices = s[0:20]
    rolling_mean = np.mean(closing_prices)
    rolling_std = np.std(closing_prices)
    
    if rolling_std == 0:
        bollinger_percent_b = 0
    else:
        bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
    
    gains = np.maximum(np.diff(closing_prices), 0)
    losses = np.maximum(-np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)) if avg_loss != 0 else 1)
    
    return np.array([bollinger_percent_b, rsi])  # Two features for mean-reversion

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    
    # Volatility-adaptive thresholds
    historical_volatility = compute_volatility(s[0:20])
    vol_threshold = historical_volatility * (0.5 if volatility_regime > 0.7 else 1.0)
    
    if abs(trend_strength) > 0.3:  # Strong trend
        new_features = compute_trend_features(s)
    elif abs(trend_strength) < 0.15:  # Sideways
        new_features = compute_mean_reversion_features(s)
    else:
        new_features = np.array([0, 0])  # Neutral features for uncertain regimes

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Regime-specific reward logic
    volatility_regime = regime_vector[1]
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0:
            reward += 50.0  # Reward for aligning with uptrend
        else:
            reward -= 20.0  # Penalty for aligning with downtrend
    elif abs(trend_strength) < 0.15:  # Sideways regime
        if enhanced_s[125] < 0.5:  # Assuming last feature is Bollinger %B
            reward += 10.0  # Reward for mean-reversion potential
        else:
            reward -= 5.0  # Penalty for chasing breakouts

    # Volatility considerations
    if volatility_regime > 0.7:  # High volatility
        reward -= 30.0  # Discourage aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds