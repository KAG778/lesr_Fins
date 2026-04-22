import numpy as np

def compute_historical_volatility(prices, window=20):
    """
    Compute historical volatility as standard deviation of returns over a specified window.
    """
    returns = np.diff(prices) / prices[:-1]  # Calculate daily returns
    if len(returns) < window:
        return np.nan  # Not enough data to compute volatility
    return np.std(returns[-window:])  # Return std of the last `window` returns

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with 125 dims base
    
    # Compute historical volatility from closing prices
    historical_volatility = compute_historical_volatility(s[0:20])
    
    # Initialize new features
    new_features = []

    if abs(trend_strength) > 0.3:  # Strong trend regime
        # Example trend-following features
        moving_average_5 = np.mean(s[0:5])  # 5-day moving average
        moving_average_20 = np.mean(s[0:20])  # 20-day moving average
        ma_crossover_distance = moving_average_5 - moving_average_20

        new_features.extend([ma_crossover_distance, historical_volatility])

    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Example mean-reversion features
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])  # Upper Bollinger band
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])  # Lower Bollinger band
        bollinger_percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger)  # Current price in Bollinger %B

        new_features.extend([bollinger_percent_b, historical_volatility])
    
    # Always include at least one additional feature
    new_features.append(np.mean(s[80:100]))  # Average volume over the last 20 days as an additional feature

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

    # Different reward logic per regime...
    if trend_strength > 0.3:  # Strong uptrend
        if momentum_signal > 0:
            reward += 20  # Positive reward for alignment
        else:
            reward -= 10  # Cautious penalty for misalignment

    elif trend_strength < -0.3:  # Strong downtrend
        reward += 5  # Mild reward for cautious actions

    elif abs(trend_strength) < 0.15:  # Sideways
        if meanrev_signal > 0:  # Price at upper Bollinger
            reward += 10  # Reward for potential mean-reversion opportunity
        elif meanrev_signal < 0:  # Price at lower Bollinger
            reward += 5

    if volatility_regime > 0.7:  # High volatility
        reward -= 15  # Penalize aggressive actions

    return reward