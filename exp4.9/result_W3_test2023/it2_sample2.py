import numpy as np

def revise_state(s, regime_vector):
    # Extract regime vector components
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    closing_prices = s[0:20]
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short-term moving average (5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long-term moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        # Use ADX as a measure of trend strength
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX calculation
        trend_consistency = np.std(daily_returns[-10:])  # Recent volatility for trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid division by zero
        rsi = (100 - (100 / (1 + np.mean(daily_returns[-14:]) / np.std(daily_returns[-14:])))
                if np.std(daily_returns[-14:]) > 0 else 0)  # Avoid division by zero
        
        range_width = np.max(closing_prices) - np.min(closing_prices)
        new_features = [bollinger_percent_b, rsi, range_width]

    # Ensure we return at least 3 new features
    new_features = np.array(new_features)
    if len(new_features) < 3:
        new_features = np.pad(new_features, (0, 3 - len(new_features)), 'constant', constant_values=np.nan)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    # Extract regime_vector from enhanced_state[120:125]
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Reward logic based on different regimes
    if trend_strength > 0.3:  # TREND regime
        reward += 50.0  # Positive reward for trend-aligned actions
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 20.0
        else:
            reward -= 10.0  # Penalize if against momentum

    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 50.0  # Negative reward for taking long positions
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 15.0  # Mild reward for shorting in downtrend

    elif -0.15 <= trend_strength <= 0.15:  # SIDEWAYS regime
        if regime_vector[3] == -1:  # Potential mean-reversion opportunity
            reward += 20.0  # Positive for counter-trend entries
        else:
            reward -= 10.0  # Penalize breakout attempts

    # Adjust reward for high volatility
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down reward magnitude

    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range