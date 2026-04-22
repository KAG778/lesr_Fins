import numpy as np

def calculate_historical_volatility(prices, window=20):
    """ Calculate historical volatility given closing prices. """
    returns = np.diff(prices) / prices[:-1]
    if len(returns) < window:
        return np.nan  # Not enough data
    return np.std(returns[-window:]) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with the base 125 dims

    # Feature list based on regime
    new_features = []
    
    closing_prices = s[0:20]
    historical_volatility = calculate_historical_volatility(closing_prices)

    if abs(trend_strength) > 0.3:  # TREND regimes
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # MA distance
        adx = np.random.rand()  # Placeholder for ADX calculation
        
        new_features.extend([ma_crossover_distance, adx, historical_volatility])
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std
        bollinger_lower = bollinger_middle - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        new_features.extend([bollinger_percent_b, historical_volatility])

    # Additional feature relevant across regimes
    new_features.append(np.mean(s[80:100]))  # Average volume over the last 20 days
    
    # Combine enhanced state with new features
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        if enhanced_s[-3] > 0:  # Assuming this is a momentum-aligned signal
            reward += 20  # Positive reward for momentum alignment
        else:
            reward -= 10  # Penalty for counter-trend action
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if enhanced_s[-4] > 0:  # Assuming this is a mean-reversion signal
            reward += 10  # Reward for mean-reversion opportunity
        elif enhanced_s[-4] < 0:  # If the signal is counter to mean-reversion
            reward -= 5  # Mild penalty for chasing breakouts

    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 15  # Penalize aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range