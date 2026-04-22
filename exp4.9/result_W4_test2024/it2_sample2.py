import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def compute_trend_features(s):
    """Compute features for trending regimes."""
    closing_prices = s[0:20]
    short_ma = np.mean(closing_prices[-5:])  # Short-term MA (5 days)
    long_ma = np.mean(closing_prices[-20:])  # Long-term MA (20 days)
    ma_crossover_distance = short_ma - long_ma

    returns = np.diff(closing_prices) / closing_prices[:-1]
    trend_consistency = np.std(returns)

    return np.array([ma_crossover_distance, trend_consistency])

def compute_mean_reversion_features(s):
    """Compute features for sideways regimes."""
    closing_prices = s[0:20]
    rolling_mean = np.mean(closing_prices)
    rolling_std = np.std(closing_prices)

    if rolling_std != 0:
        bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
    else:
        bollinger_percent_b = 0
    
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100

    return np.array([bollinger_percent_b, rsi])

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    historical_volatility = compute_volatility(s[0:20])

    if abs(trend_strength) > 0.3:  # TREND regime
        new_features = compute_trend_features(s)
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        new_features = compute_mean_reversion_features(s)
    else:
        new_features = np.array([0, 0])  # Neutral features for uncertain regimes

    # Include historical volatility as a feature
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
        return -100.0  # Strong negative in crisis

    # Determine reward based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Reward for positive momentum
        else:  # Downtrend
            reward -= 25.0  # Penalty for counter-trend entries
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        bollinger_percent_b = enhanced_s[125]  # Assuming this is the Bollinger %B feature
        if bollinger_percent_b < 0.5:  # Mean-reversion condition
            reward += 10.0  # Positive reward for mean-reversion
        else:
            reward -= 5.0  # Penalize for chasing breakouts

    # Volatility adjustment
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 15.0  # Penalize aggressive entries
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds