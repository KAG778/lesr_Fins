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
    historical_volatility = calculate_historical_volatility(s[0:20])

    # Initialize new features
    new_features = []

    # TREND regimes
    if abs(trend_strength) > 0.3:  
        moving_average_short = np.mean(s[0:5])  # 5-day moving average
        moving_average_long = np.mean(s[0:20])  # 20-day moving average
        ma_crossover_distance = moving_average_short - moving_average_long
        
        new_features.extend([ma_crossover_distance, historical_volatility])

    # SIDEWAYS regimes
    elif abs(trend_strength) < 0.15:  
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])  # Upper Bollinger band
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])  # Lower Bollinger band
        bollinger_percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        new_features.extend([bollinger_percent_b, historical_volatility])

    # Always include average volume as an additional feature
    average_volume = np.mean(s[80:100])  # Assuming volume data is in indices 80-100
    new_features.append(average_volume)

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
        return -100.0  # Strong negative for any buying action during crisis

    # Adjust reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        if trend_strength > 0:  # Uptrend
            reward += 20  # Reward for buying in an uptrend
        else:  # Downtrend
            reward -= 20  # Penalize for buying in a downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if enhanced_s[-4] > 0:  # Assuming -4 is a mean-reversion signal
            reward += 10  # Reward for mean-reversion opportunity
        else:
            reward -= 5  # Mild penalty for chasing breakouts

    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 15  # Penalize aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range