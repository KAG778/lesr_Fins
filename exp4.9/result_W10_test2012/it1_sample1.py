import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    # Calculate closing prices for the last 20 days
    closing_prices = s[0:20]
    
    # Calculate moving averages
    short_ma = np.mean(closing_prices[-5:])  # last 5 days MA
    long_ma = np.mean(closing_prices)  # last 20 days MA
    ma_crossover_distance = short_ma - long_ma
    
    # Calculate daily returns and historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    # Adding features based on regimes
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        new_features.append(ma_crossover_distance)  # Moving Average Crossover Distance
        new_features.append(historical_volatility)  # Historical volatility
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        new_features.append(bollinger_percent_b)  # Bollinger %B
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width
    # Additional regime-specific features can be incorporated here if necessary
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative reward for any entries in crisis
    
    # Reward logic based on regimes
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Strong positive for aligning with uptrend
        else:  # Downtrend
            reward -= 30.0  # Cautious negative for aligning with downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 20.0  # Mild positive for mean-reversion opportunity
        else:
            reward -= 10.0  # Penalize breakout chases

    # Adjust reward based on volatility regime
    if regime_vector[1] > 0.7:  # HIGH VOLATILITY
        reward *= 0.5  # Scale down reward magnitudes
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range