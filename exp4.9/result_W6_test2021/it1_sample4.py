import numpy as np

def calculate_historical_volatility(prices, window=20):
    """ Calculate historical volatility given closing prices. """
    returns = np.diff(prices) / prices[:-1]
    if len(returns) < window:
        return np.nan  # Not enough data to compute volatility
    return np.std(returns[-window:]) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with 125 dims base
    
    # Calculate historical volatility from closing prices
    closing_prices = s[0:20]  # Assuming prices are in the first 20 elements
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    # Initialize new features
    new_features = []

    if abs(trend_strength) > 0.3:  # TREND regime
        # Add trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # Distance between MAs
        
        new_features.extend([ma_crossover_distance, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Add mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std  # Upper Bollinger band
        bollinger_lower = bollinger_middle - 2 * bollinger_std  # Lower Bollinger band
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        
        new_features.extend([bollinger_percent_b, historical_volatility])
        
    # Always include average volume as an additional feature
    average_volume = np.mean(s[80:100])  # Assuming volume data is in indices 80-100
    new_features.append(average_volume)

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
        # Strong negative reward for BUY signals
        return -100.0  # Strong penalty for any buying action during crisis

    # Different reward logic per regime
    if trend_strength > 0.3:  # TREND up
        if momentum_signal > 0:  # Aligned with momentum
            reward += 20  # Positive reward
        else:
            reward -= 10  # Penalty for misalignment
    
    elif trend_strength < -0.3:  # TREND down
        reward += 5  # Cautious reward for trend-following
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if meanrev_signal > 0:  # Price at upper Bollinger band
            reward += 15  # Positive reward for potential mean-reversion
        elif meanrev_signal < 0:  # Price at lower Bollinger band
            reward += 5  # Mild reward

    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 20  # Strong penalty to discourage aggressive positions
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range