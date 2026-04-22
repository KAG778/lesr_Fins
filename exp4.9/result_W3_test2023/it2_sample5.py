import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    if abs(trend_strength) > 0.3:  # TREND regime
        short_ma = np.mean(closing_prices[-5:])  # Short-term moving average
        long_ma = np.mean(closing_prices[-20:])  # Long-term moving average
        ma_crossover_distance = short_ma - long_ma
        
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX
        trend_consistency = np.std(daily_returns[-10:])  # Recent trend consistency

        new_features = [ma_crossover_distance, adx, trend_consistency, historical_volatility]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid division by zero
        rsi = (100 - (100 / (1 + np.mean(daily_returns[-14:]) / np.std(daily_returns[-14:])) 
              if np.std(daily_returns[-14:]) > 0 else 0))  # Avoid division by zero

        price_range = np.max(closing_prices) - np.min(closing_prices)

        new_features = [bollinger_percent_b, rsi, price_range]

    # Ensure we return at least 3 features
    new_features += [0] * (3 - len(new_features))  # Fill with zeros if necessary
    new_features = np.array(new_features)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Reward logic based on different regimes
    if trend_strength > 0.3:  # TREND UP
        reward += 50.0  # Base reward for trend-aligned actions
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 20.0
        elif regime_vector[2] < 0:  # Weak momentum
            reward -= 10.0

    elif trend_strength < -0.3:  # TREND DOWN
        reward -= 25.0  # Negative reward for buying in a downtrend
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 10.0  # Encourage shorts in downtrend

    elif -0.15 <= trend_strength <= 0.15:  # SIDEWAYS market
        if regime_vector[3] == -1:  # Potential mean-reversion opportunity
            reward += 20.0  # Positive for counter-trend entries

    # Adjust reward for volatility regime
    volatility_regime = regime_vector[1]
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down rewards in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range