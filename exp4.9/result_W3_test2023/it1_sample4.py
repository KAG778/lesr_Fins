import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    closing_prices = s[0:20]
    
    # Calculate historical volatility (e.g., using the last 20 days closing prices)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    # Define thresholds for feature selection based on volatility
    volatility_threshold = np.mean(historical_volatility) + 0.5 * np.std(historical_volatility)

    if abs(trend_strength) > 0.3:  # Strong trend
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short-term moving average (5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long-term moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX
        trend_consistency = np.std(daily_returns[-10:])  # Recent volatility for trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency]
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid div by zero
        rsi = np.mean(daily_returns[-14:])  # Simplified RSI calculation based on returns
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]

    # Ensure at least 3 features are included
    new_features = np.array(new_features)
    if len(new_features) < 3:
        new_features = np.pad(new_features, (0, 3 - len(new_features)), 'constant', constant_values=np.nan)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Regime-based reward logic
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50.0  # Positive reward for trend-aligned actions
        if regime_vector[2] > 0:  # Strong momentum
            reward += 20.0  # Reward for positive momentum
        else:
            reward -= 10.0  # Cautious if momentum is against the trend

    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 25.0  # Negative reward for buying in downtrend
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 10.0  # Reward for aligning with strong downward momentum

    elif -0.15 <= trend_strength <= 0.15:  # Sideways market
        reward += 10.0  # Mild positive for holding
        if regime_vector[3] == -1:  # Mean-reversion opportunity
            reward += 20.0  # Positive for potential bounce at lower Bollinger

    # Adjust reward for volatility regime
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Scale down rewards in high volatility to discourage aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range