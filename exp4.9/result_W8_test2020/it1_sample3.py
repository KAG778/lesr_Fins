import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Base 125 dimensions
    new_features = []

    # Extract price-related data
    closing_prices = s[0:20]
    if len(closing_prices) < 20:
        return enhanced  # Not enough data for feature computation

    # Calculate daily returns
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    
    # Feature calculations based on market regime
    if abs(trend_strength) > 0.3:  # Trend regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days
        long_ma = np.mean(closing_prices[-20:])   # Last 20 days
        new_features.append(short_ma - long_ma)  # MA Crossover Distance
        
        adx = np.std(daily_returns[-14:]) * 100  # Placeholder for ADX-like feature
        new_features.append(adx)
    else:  # Sideways regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.append(bollinger_percent_b)
        new_features.append(rsi)

    # Volatility Feature: Average True Range (ATR)
    tr = np.maximum(s[40:60][1:] - s[60:80][1:], 
                    np.maximum(np.abs(s[40:60][1:] - closing_prices[:-1]), 
                               np.abs(s[60:80][1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    new_features.append(atr)

    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative for any entry during a crisis

    # Regime-specific reward logic
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0:  # Uptrend
            reward += 20  # Base reward for trend-following
            if regime_vector[2] > 0:  # Positive momentum
                reward += 20  # Additional reward for momentum alignment
        else:  # Downtrend
            reward -= 10  # Cautious for downtrend
    elif abs(trend_strength) < 0.15:  # Sideways regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 15  # Positive for mean-reversion
        else:
            reward -= 10  # Caution for breakouts
        
    # Penalize for high volatility
    if volatility_regime > 0.7:
        reward -= 20  # Strong penalty for aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds