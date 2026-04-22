import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]

    # Use historical standard deviation for adaptive thresholds
    historical_std = np.std(daily_returns)

    if abs(trend_strength) > 0.3:  # TREND regime
        # Calculate trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days MA
        long_ma = np.mean(closing_prices)  # Last 20 days MA
        ma_crossover = short_ma - long_ma
        
        # Historical volatility
        historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        
        new_features.append(ma_crossover)
        new_features.append(historical_volatility)
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Calculate mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * historical_std
        lower_bollinger = np.mean(closing_prices) - 2 * historical_std
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        new_features.append(bollinger_percent_b)
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width
        
    # Append new features to the enhanced state
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
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Strong uptrend
            reward += 50.0  # Reward for aligning with trend
        elif trend_strength < 0:  # Strong downtrend
            reward -= 30.0  # Cautious reward for downtrend
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 20.0  # Reward for mean-reversion opportunity
        else:
            reward -= 10.0  # Penalize for chasing breakouts
    
    # Adjust reward based on volatility regime
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down rewards in high volatility
        
    return np.clip(reward, -100, 100)  # Ensure reward is within range