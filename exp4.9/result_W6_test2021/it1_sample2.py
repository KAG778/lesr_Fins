import numpy as np

def calculate_historical_volatility(prices):
    """Calculate the historical volatility given closing prices."""
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # Base 125 dims
    
    # Initialize new features
    new_features = []
    
    closing_prices = s[0:20]
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    # Feature engineering based on current regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short-term MA
        moving_average_long = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = moving_average_short - moving_average_long  # Distance between MAs
        
        new_features.extend([ma_crossover_distance, historical_volatility])
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        bollinger_middle = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_middle + 2 * bollinger_std
        bollinger_lower = bollinger_middle - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        new_features.extend([bollinger_percent_b, historical_volatility])
        
    # Include average volume over the last 20 days as an additional feature
    new_features.append(np.mean(s[80:100]))  # Average volume
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative for entries in crisis

    # Reward logic based on the current regime
    if trend_strength > 0.3:  # TREND UP regime
        if enhanced_s[-3] > 0:  # Assuming -3 is an aligned trend-following signal
            reward += 20  # Positive reward for aligned trend
        else:
            reward -= 10  # Penalty for misalignment
    
    elif trend_strength < -0.3:  # TREND DOWN regime
        reward += 5  # Mild reward for cautious actions
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[-4] < 0:  # Assuming -4 is a mean-reversion signal
            reward += 10  # Reward for mean-reversion opportunity
        
    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH VOLATILITY
        reward -= 15  # Penalize aggressive actions
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range