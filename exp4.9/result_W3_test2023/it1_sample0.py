import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []
    
    closing_prices = s[0:20]
    
    # Historical volatility for adaptive thresholds
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        long_ma = np.mean(closing_prices[-20:])   # Long MA (last 20 days)
        ma_crossover_distance = short_ma - long_ma
        adx = np.mean(np.abs(daily_returns[-5:]))  # Placeholder for ADX
        
        new_features = [ma_crossover_distance, adx, historical_volatility]
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)  # Avoid div by zero
        rsi = np.mean(daily_returns[-14:])  # Placeholder for RSI
        range_width = np.max(closing_prices) - np.min(closing_prices)
        
        new_features = [bollinger_percent_b, rsi, range_width]
    
    # Ensure we return at least 3 new features
    new_features = new_features + [0]*(3 - len(new_features))  # Fill with zeros if necessary
    
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on different regimes
    if trend_strength > 0.3:  # TREND UP
        reward += 50.0  # Positive reward for trend-aligned actions
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 20.0
        elif regime_vector[2] < 0:  # Strong downward momentum
            reward -= 10.0
            
    elif trend_strength < -0.3:  # TREND DOWN
        reward -= 25.0  # Negative reward for trend-aligned actions
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 10.0  # Mild positive for aligning with downward momentum
            
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        reward += 10.0  # Mild positive for holding
        if regime_vector[3] == -1:  # Price at lower Bollinger
            reward += 20.0  # Good opportunity to bounce
        elif regime_vector[3] == 1:  # Price at upper Bollinger
            reward -= 10.0  # Cautious for mean reversion in a downtrend

    # Adjust reward for volatility regime
    volatility_regime = regime_vector[1]
    if volatility_regime > 0.7:  # High volatility
        reward *= 0.5  # Scale down reward magnitude
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]