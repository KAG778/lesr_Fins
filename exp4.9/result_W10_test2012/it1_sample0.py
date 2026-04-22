import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate closing prices for the last 20 days
    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]

    # Use volatility to adapt thresholds
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    vol_threshold = historical_volatility * 0.5  # Adjusted threshold based on volatility

    # Regime-specific feature engineering
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # last 5 days MA
        long_ma = np.mean(closing_prices)  # last 20 days MA
        ma_crossover_distance = short_ma - long_ma
        
        # Feature capturing trend momentum
        momentum_signal = np.mean(daily_returns[-5:])  # Average momentum over last 5 days
        new_features.append(ma_crossover_distance)
        new_features.append(momentum_signal)
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        # Capture price range for the sideways market
        price_range = np.max(closing_prices) - np.min(closing_prices)
        new_features.append(bollinger_percent_b)
        new_features.append(price_range)
    
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
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Strong uptrend
            reward += 50.0  # Reward for uptrend alignment
        else:  # Strong downtrend
            reward -= 30.0  # Penalty for downtrend
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 20.0  # Reward for mean-reversion opportunities
        else:
            reward -= 10.0  # Penalty for chasing breakouts

    # Adjust reward based on volatility regime
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down rewards
    
    return np.clip(reward, -100, 100)  # Ensure reward is within range