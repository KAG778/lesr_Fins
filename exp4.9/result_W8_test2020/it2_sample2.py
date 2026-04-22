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
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate daily returns to assist in feature calculations
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    
    # Historical volatility for adaptive thresholds
    historical_volatility = np.std(closing_prices)

    # TREND regime (|trend_strength| > 0.3)
    if abs(trend_strength) > 0.3:
        short_sma = np.mean(closing_prices[-5:])  # Short-term SMA
        long_sma = np.mean(closing_prices[-20:])   # Long-term SMA
        sma_crossover_distance = short_sma - long_sma
        
        # ADX-like feature for trend strength
        directional_movement = np.mean(high_prices[-5] - low_prices[-5]) / np.mean(closing_prices[-5])
        adx = directional_movement * 100  # Scale to percentage
        
        new_features.extend([sma_crossover_distance, adx, historical_volatility])

    # SIDEWAYS regime (|trend_strength| < 0.15)
    elif abs(trend_strength) < 0.15:
        # Bollinger Bands %B
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        
        # RSI calculation for mean-reversion
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.extend([bollinger_percent_b, rsi, historical_volatility])

    # Append Average True Range (ATR) for volatility adaptation
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

    # TREND regime (|trend_strength| > 0.3)
    if abs(trend_strength) > 0.3:
        if trend_strength > 0:  # Uptrend
            reward += 30  # Strong positive reward for trend-following
            if momentum_signal > 0:  # Strong momentum
                reward += 10  # Additional reward for momentum alignment
        else:  # Downtrend
            reward -= 10  # Cautious penalty for counter-trend

    # SIDEWAYS regime (|trend_strength| < 0.15)
    elif abs(trend_strength) < 0.15:
        if meanrev_signal > 0:  # Mean-reversion opportunity
            reward += 15  # Positive for mean-reversion
        else: 
            reward -= 5  # Penalize for chasing breakouts

    # HIGH VOLATILITY regime
    if volatility_regime > 0.7:
        reward -= 20  # Severe penalty for aggressive positions in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds