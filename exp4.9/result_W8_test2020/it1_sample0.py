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

    # Calculate daily returns to aid in feature calculations
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]

    # Feature logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_sma = np.mean(closing_prices[-5:])  # Short-term SMA
        long_sma = np.mean(closing_prices[-20:])   # Long-term SMA
        sma_crossover_distance = short_sma - long_sma
        
        # Average True Range (ATR) for assessing volatility
        atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                                  np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                             np.abs(low_prices[1:] - closing_prices[:-1])))
                      ) if len(closing_prices) > 1 else 0
        
        new_features.extend([sma_crossover_distance, atr, np.std(daily_returns)])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        
        # Bollinger Bands %B
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        
        # Relative Strength Index (RSI)
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.extend([bollinger_percent_b, rsi, np.max(closing_prices) - np.min(closing_prices)])

    # Add new features to the enhanced state
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Different reward logic per regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend with positive momentum
            reward += 20  # Positive reward for trend-following
        elif trend_strength < 0:  # Downtrend
            reward -= 10  # Cautious reward for downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 15  # Positive for mean-reversion
        else:
            reward -= 10  # Penalize for chasing breakouts

    # Adjust for HIGH VOLATILITY
    if volatility_regime > 0.7:
        reward -= 15  # Penalize aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds