import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate closing prices for the last 20 days
    closing_prices = s[0:20]
    
    # Calculate moving averages
    short_ma = np.mean(closing_prices[-5:])  # last 5 days MA
    long_ma = np.mean(closing_prices)  # last 20 days MA
    ma_crossover_distance = short_ma - long_ma
    
    # Calculate volatility based on closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    # Adding trend-following features if in strong trend
    if abs(trend_strength) > 0.3:
        new_features.append(ma_crossover_distance)  # Moving Average Crossover Distance
        new_features.append(historical_volatility)  # Historical volatility
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
    else:
        # Adding mean-reversion features if in sideways market
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        new_features.append(bollinger_percent_b)  # Bollinger %B
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width
        
    # Ensure new features are concatenated to the enhanced state
    return np.concatenate([enhanced, new_features])


def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on regimes
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50.0  # Positive reward for aligning with trend
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 30.0  # Cautious reward for downtrend
    else:  # Sideways market
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 20.0  # Mild positive for mean-reversion opportunity
    
    # Adjust reward based on volatility regime
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range