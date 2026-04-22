import numpy as np

def calculate_atr(prices, period=14):
    # Calculate Average True Range (ATR)
    high_low = prices[2] - prices[3]  # high - low
    high_close = np.abs(prices[2] - prices[1])  # high - previous close
    low_close = np.abs(prices[3] - prices[1])  # low - previous close
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return np.mean(tr[-period:])  # ATR for the last 'period' days

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate historical volatility
    daily_returns = np.diff(s[0:20]) / s[0:19]  # Closing prices returns
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    
    # Compute features based on regime
    if abs(trend_strength) > 0.3:
        # Trend-following features
        short_ma = np.mean(s[0:20][-5:])  # Short MA (5 days)
        long_ma = np.mean(s[0:20][-20:])  # Long MA (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX
        trend_consistency = np.sum(daily_returns > 0) / len(daily_returns)  # Trend consistency
        
        new_features = [ma_crossover_distance, adx, trend_consistency]

    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.mean(daily_returns[-14:] > 0)  # Simplified RSI calculation
        
        new_features = [bollinger_percent_b, rsi, np.ptp(s[0:20])]  # Range width

    if volatility_regime > 0.7:
        # High volatility features
        atr = calculate_atr(s[40:60].reshape(20, 1))  # High prices
        new_features.append(atr)

    if crisis_signal > 0.5:
        # Crisis features
        # We might want to include drawdown or max consecutive losses
        new_features = [-1] * 3  # Placeholder for crisis signals
        
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on regime
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 20.0  # Positive reward
        elif regime_vector[2] < 0:  # Strong downward momentum
            reward -= 10.0  # Cautious reward

    elif trend_strength < -0.3:  # Strong downtrend
        if regime_vector[2] < 0:  # Strong downward momentum
            reward += 10.0  # Cautious positive reward
        elif regime_vector[2] > 0:  # Strong upward momentum
            reward -= 20.0  # Negative reward

    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward += 10.0  # Mild positive for counter-trend entry
        elif regime_vector[3] > 0:  # Price at upper Bollinger
            reward -= 10.0  # Mild negative for counter-trend entry

    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5.0  # Negative for aggressive entries

    return reward