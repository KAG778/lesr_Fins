import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    closing_prices = s[0:20]
    historical_volatility = compute_volatility(closing_prices)

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days
        long_ma = np.mean(closing_prices[-20:])  # Last 20 days
        ma_crossover_distance = short_ma - long_ma
        
        momentum = (closing_prices[-1] - closing_prices[-2]) / historical_volatility if historical_volatility != 0 else 0

        new_features = [ma_crossover_distance, momentum, historical_volatility]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std) if rolling_std != 0 else 0
        
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
        
        range_width = np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])
        new_features = [bollinger_percent_b, rsi, range_width]

    # Include historical volatility as a feature
    new_features.append(historical_volatility)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # TREND regime
    if abs(trend_strength) > 0.3:
        if trend_strength > 0:  # Uptrend
            reward += 50.0 * trend_strength  # Reward positive momentum entries
        else:  # Downtrend
            reward -= 20.0 * abs(trend_strength)  # Penalize aggressive buys

    # SIDEWAYS regime
    elif abs(trend_strength) < 0.15:
        if enhanced_s[125] < 0.5:  # Assuming last feature is Bollinger %B
            reward += 10  # Positive for mean-reversion potential
        else:
            reward -= 5  # Penalize for chasing breakouts

    # High volatility regime
    if volatility_regime > 0.7:
        reward -= 15  # Scale down rewards for aggressive positions
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds