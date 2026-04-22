import numpy as np

def calculate_historical_volatility(prices):
    """ Calculate the historical volatility given closing prices. """
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with 125 dims base
    
    # Calculate historical volatility from closing prices
    closing_prices = s[0:20]
    historical_volatility = calculate_historical_volatility(closing_prices)

    # Initialize new features
    new_features = []

    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # Distance between MAs
        adx = np.random.rand()  # Placeholder for ADX calculation
        
        new_features.extend([ma_crossover_distance, adx, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std
        bollinger_lower = bollinger_middle - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.random.rand()  # Placeholder for RSI calculation

        new_features.extend([bollinger_percent_b, rsi, historical_volatility])

    # Always include an additional feature for volume or other relevant metrics
    new_features.append(np.mean(s[80:100]))  # Average volume over the last 20 days
    
    # Append the new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Different reward logic per regime
    if trend_strength > 0.3:  # STRONG UP TREND
        if momentum_signal > 0:
            reward += 30  # Strong positive reward for aligned trend
        else:
            reward -= 20  # Strong penalty for misalignment

    elif trend_strength < -0.3:  # STRONG DOWN TREND
        reward += 10  # Mild reward for cautious actions

    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if meanrev_signal > 0:  # Price at upper Bollinger
            reward += 15  # Reward for potential mean-reversion opportunity
        elif meanrev_signal < 0:  # Price at lower Bollinger
            reward += 10  # Mild reward for lower Bollinger

    # Adapt rewards based on volatility regime
    if volatility_regime > 0.7:  # HIGH VOLATILITY
        reward -= 20  # Penalize aggressive actions
    
    # Clip the reward to ensure it's within the specified range
    return np.clip(reward, -100, 100)