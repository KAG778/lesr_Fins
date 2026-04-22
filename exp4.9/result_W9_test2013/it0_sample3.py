import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate the average price over the last 20 days
    avg_price = np.mean(s[0:20])  # Closing prices
    last_price = s[0]  # Latest closing price
    price_change = (last_price - avg_price) / avg_price if avg_price != 0 else 0  # Price change relative to average

    # Calculate volatility based on the last 20 days of closing prices
    daily_returns = np.diff(s[0:20]) / s[0:19]
    historical_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0.01  # Avoid division by zero

    if abs(trend_strength) > 0.3:  # Strong Trend
        # Trend-following features
        ma_short = np.mean(s[0:10])  # Short MA
        ma_long = np.mean(s[10:20])  # Long MA
        ma_crossover = (ma_short - ma_long) / (ma_long if ma_long != 0 else 1)  # MA crossover distance
        adx = np.mean(np.abs(daily_returns)) / (historical_volatility if historical_volatility != 0 else 1)  # ADX approximation
        new_features.extend([ma_crossover, adx, price_change])
    elif abs(trend_strength) < 0.15:  # Sideways
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (last_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) != 0 else 0
        rsi = np.mean(daily_returns)  # Simplified RSI calculation
        new_features.extend([bollinger_percent_b, rsi, price_change])
    
    # High Volatility
    if volatility_regime > 0.7:
        atr = np.mean(np.abs(np.diff(s[0:20])))  # Average True Range approximation
        new_features.append(atr)

    # Crisis regime
    if crisis_signal > 0.5:
        max_consecutive_losses = 0  # Placeholder for logic
        new_features.append(max_consecutive_losses)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Reward logic based on regimes
    if trend_strength > 0.3:
        if regime_vector[2] > 0:  # Strong upward momentum
            reward = 50.0
        elif regime_vector[2] < 0:  # Strong downward momentum
            reward = -10.0  # Cautious reward
    elif trend_strength < -0.3:
        reward = -20.0  # Strong downtrend
    else:
        # Sideways market
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward = 20.0  # Positive for mean-reversion entry
        elif regime_vector[3] > 0:  # Price at upper Bollinger
            reward = -10.0  # Cautious for mean-reversion entry

    # High volatility impact
    if regime_vector[1] > 0.7:
        reward -= 20.0  # Penalize rewards in high volatility

    return reward