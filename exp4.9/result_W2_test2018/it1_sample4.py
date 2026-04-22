import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state with raw state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions
    new_features = []

    # Calculate historical volatility
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices)  # 20-day standard deviation
    
    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short moving average (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long moving average (last 20 days)
        ma_crossover_distance = short_ma - long_ma

        # Average True Range (ATR) for volatility measures
        atr = np.mean(np.abs(np.diff(closing_prices)))  # Simple ATR approximation
        
        new_features.extend([ma_crossover_distance, atr, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + (bollinger_std * 2)
        lower_bollinger = bollinger_mid - (bollinger_std * 2)
        bollinger_pct_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        rsi = np.mean(np.diff(closing_prices) > 0)  # Simple RSI calculation
        
        new_features.extend([bollinger_pct_b, rsi, historical_volatility])

    if volatility_regime > 0.7:  # HIGH_VOL regimes
        # Include signals that indicate high volatility
        new_features.append(historical_volatility)  # Historical volatility as a feature

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]

    reward = 0.0
    
    # **CRITICAL**: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # TREND regime reward logic
    if abs(trend_strength) > 0.3:
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Positive reward for trend-following
            if regime_vector[2] < 0:  # Negative momentum
                reward -= 30.0  # Penalize against trend
        else:  # Downtrend
            reward -= 30.0  # Cautious reward for downtrending market
            if regime_vector[2] > 0:  # Positive momentum
                reward -= 50.0  # Penalize against trend

    # SIDEWAYS regime reward logic
    elif abs(trend_strength) < 0.15:
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward += 20.0  # Positive for counter-trend entries
        elif regime_vector[3] > 0:  # Price at upper Bollinger
            reward -= 20.0  # Penalize breakout chases

    # HIGH_VOL regime adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 10.0  # Penalize aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]