import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dimensions
    new_features = []

    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate daily returns for momentum features and volatility adaptation
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    
    # Obtain historical volatility
    historical_volatility = np.std(closing_prices)

    # Feature logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_sma = np.mean(closing_prices[-5:])  # Last 5 days
        long_sma = np.mean(closing_prices[-20:])   # Last 20 days
        sma_crossover_distance = short_sma - long_sma
        
        adx = np.std(daily_returns[-14:]) * 100  # Placeholder for ADX-like feature
        new_features.extend([sma_crossover_distance, adx, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        
        # RSI calculation
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.extend([bollinger_percent_b, rsi, historical_volatility])

    # Calculate ATR for volatility adaptation
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
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

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend with positive momentum
            reward += 30  # Strong positive reward for trend-following
        elif trend_strength < 0:  # Downtrend
            reward -= 10  # Cautious penalty for counter-trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Strong mean-reversion opportunity
            reward += 15  # Positive for mean-reversion
        else:
            reward -= 5  # Cautious for breakout attempts
        
    # Penalize for high volatility
    if volatility_regime > 0.7:
        reward -= 20  # Severe penalty for aggressive positions in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds