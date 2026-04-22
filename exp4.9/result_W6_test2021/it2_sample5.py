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
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    
    # Calculate historical volatility from closing prices
    closing_prices = s[0:20]
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    # Initialize new features
    new_features = []
    
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
        bollinger_upper = bollinger_middle + 2 * bollinger_std  # Upper Bollinger band
        bollinger_lower = bollinger_middle - 2 * bollinger_std  # Lower Bollinger band
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        new_features.extend([bollinger_percent_b, historical_volatility])
    
    # Always include average volume over the last 20 days as an additional feature
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
        return -100.0  # Strong penalty for any buying action during crisis

    # Volatility adaptive thresholds
    volatility_threshold = 0.5 + (volatility_regime * 0.5)  # Adjusting threshold based on volatility

    # Reward logic based on the current regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        if trend_strength > 0:  # Uptrend
            reward += 20  # Reward for aligned momentum
        else:  # Downtrend
            reward -= 20  # Penalty for buying in a downtrend
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if enhanced_s[-3] > 0:  # Assuming this is a mean-reversion signal
            reward += 10  # Reward for potential mean-reversion opportunity
        else:
            reward -= 5  # Penalty for misalignment with mean-reversion
            
    # Adjust for high volatility
    if volatility_regime > volatility_threshold:  # HIGH VOL regimes
        reward -= 15  # Penalize aggressive actions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range