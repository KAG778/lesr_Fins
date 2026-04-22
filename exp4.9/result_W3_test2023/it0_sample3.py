import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Start with raw state and regime vector
    
    # Compute new features based on the market regime
    new_features = []
    
    # Calculate historical volatility from closing prices
    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    if abs(trend_strength) > 0.3:  # Strong trend
        # Trend-following features
        # Example features: Moving average distance (short MA - long MA), ADX (placeholders)
        short_ma = np.mean(closing_prices[-5:])  # Short-term moving average (5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long-term moving average (20 days)
        ma_distance = short_ma - long_ma
        
        # Placeholder ADX calculation (need price data for actual calculation)
        adx = np.random.rand()  # Replace with actual ADX calculation
        
        new_features = [ma_distance, adx, historical_volatility]
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        # Mean-reversion features
        # Example features: Bollinger %B, RSI, range width
        price_range = np.max(closing_prices) - np.min(closing_prices)
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + (bollinger_std * 2)
        bollinger_lower = bollinger_mid - (bollinger_std * 2)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        
        rsi = np.random.rand()  # Replace with actual RSI calculation
        
        new_features = [bollinger_percent_b, price_range, rsi]
    
    # Ensure we return at least 3 new features
    new_features += [0, 0]  # Fill with placeholders if necessary
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    if trend_strength > 0.3:  # Strong uptrend
        reward += 50.0  # Positive reward for trend-aligned actions
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 25.0  # Cautious negative reward for downtrend
    
    # Reward adjustments based on momentum and mean reversion signals
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    
    if momentum_signal > 0 and trend_strength > 0.3:
        reward += 20.0  # Strong upward momentum in uptrend
    elif momentum_signal < 0 and trend_strength < -0.3:
        reward += 10.0  # Strong downward momentum in downtrend
    
    if meanrev_signal == 1:  # Price at upper Bollinger
        reward -= 15.0  # Cautious for mean-reversion in uptrend
    elif meanrev_signal == -1:  # Price at lower Bollinger
        reward += 15.0  # Positive for potential bounce in downtrend
    
    # Adjust reward for volatility regime
    volatility_regime = regime_vector[1]
    if volatility_regime > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude in high volatility
    
    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range