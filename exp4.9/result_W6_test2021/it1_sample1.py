import numpy as np

def calculate_historical_volatility(prices):
    """ Calculate the historical volatility given closing prices. """
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with 125 dims base
    new_features = []
    
    closing_prices = s[0:20]
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    # Regime-specific features
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # Distance between MAs
        adx = np.random.rand()  # Placeholder for ADX calculation (consider implementing ADX)

        new_features.extend([ma_crossover_distance, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std
        bollinger_lower = bollinger_middle - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B

        new_features.extend([bollinger_percent_b, historical_volatility])

    if volatility_regime > 0.7:  # Incorporate volatility as a feature
        new_features.append(np.mean(closing_prices))  # Current price as additional feature in high volatility

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

    # Reward logic by regime
    if trend_strength > 0.3:  # TREND UP
        reward += 20  # Base reward for trend-following
        if enhanced_s[-3] < 0:  # If momentum is negative
            reward -= 15  # Penalize for counter-trend actions

    elif trend_strength < -0.3:  # TREND DOWN
        reward += 10  # Reward cautious actions in downtrend
        if enhanced_s[-3] > 0:  # If momentum is positive
            reward -= 10  # Penalize for counter-trend actions

    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if enhanced_s[-4] > 0:  # If mean-reversion signal is positive
            reward += 10  # Reward for mean-reversion opportunities
        else:
            reward -= 5  # Penalize for chasing breakouts

    # Adjust for high volatility
    if volatility_regime > 0.7:
        reward -= 20  # Scale down all rewards to discourage aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range