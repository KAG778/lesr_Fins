import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dimensions
    new_features = []

    # State representation based on regime
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Historical volatility
    historical_volatility = np.std(closing_prices)  # Standard deviation of closing prices

    # Use volatility-adaptive thresholds
    volatility_threshold = historical_volatility * 0.5  # Adjust as necessary

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short-term MA
        long_ma = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = short_ma - long_ma
        new_features.append(ma_crossover_distance)
        
        # ADX-like feature for trend strength
        directional_movement = np.mean(high_prices[-5] - low_prices[-5]) / np.mean(closing_prices[-5])
        adx = directional_movement * 100  # Scale to percentage
        new_features.append(adx)

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
        new_features.append(bollinger_percent_b)

        # RSI calculation
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        new_features.append(rsi)

    # Calculate ATR for volatility adaptation
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    new_features.append(atr)

    # Append historical volatility
    new_features.append(historical_volatility)

    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 20  # Base reward for trend-following
            if regime_vector[2] > 0:  # Strong upward momentum
                reward += 10  # Additional reward for momentum alignment
        else:  # Downtrend
            reward -= 10  # Penalize for being in a downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 15  # Mild positive for mean-reversion
        else:  # Penalize breakout chases
            reward -= 5

    # Adjust for HIGH_VOL regime
    if volatility_regime > 0.7:
        reward -= 15  # Penalize for aggressive entries during high volatility
    
    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds