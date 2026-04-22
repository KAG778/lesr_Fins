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
    
    # Calculate historical volatility from closing prices
    historical_volatility = calculate_historical_volatility(s[0:20])

    # Feature list based on trend strength
    new_features = []

    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        moving_average_5 = np.mean(s[0:5])  # 5-day moving average
        moving_average_20 = np.mean(s[0:20])  # 20-day moving average
        ma_crossover_distance = moving_average_5 - moving_average_20

        new_features.extend([ma_crossover_distance, historical_volatility])
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])  # Upper Bollinger band
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])  # Lower Bollinger band
        bollinger_percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger)  # %B

        new_features.extend([bollinger_percent_b, historical_volatility])
    
    # Always include average volume over the last 20 days as an additional feature
    new_features.append(np.mean(s[80:100]))  # Average volume over the last 20 days
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative reward for entries during crisis

    # Adjust reward based on regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        if trend_strength > 0:  # Uptrend
            reward += 20  # Reward for buying in an uptrend
        else:  # Downtrend
            reward -= 20  # Penalize for buying in a downtrend
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if enhanced_s[-3] > 0:  # Example: if mean-reversion signal is positive
            reward += 10  # Reward for potential mean-reversion opportunity
        else:
            reward -= 5  # Penalize for misalignment with mean-reversion
            
    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH VOL regimes
        reward -= 15  # Penalize aggressive actions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range