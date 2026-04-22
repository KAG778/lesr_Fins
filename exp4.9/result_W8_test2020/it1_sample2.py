import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Extract closing prices and other relevant data
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate daily returns
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]

    # Feature calculations based on regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        short_sma = np.mean(closing_prices[-5:])  # Short-term SMA
        long_sma = np.mean(closing_prices[-20:])   # Long-term SMA
        sma_crossover_distance = short_sma - long_sma
        
        # ADX-like feature
        adx = np.std(daily_returns[-14:]) * 100  # Placeholder for trend strength
        
        new_features.extend([sma_crossover_distance, adx])
        
        # Volatility condition
        if volatility_regime > 0.7:
            new_features.append(np.std(closing_prices))  # Add volatility measure

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        if rolling_std > 0:
            bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
        else:
            bollinger_percent_b = 0.0
            
        new_features.append(bollinger_percent_b)

        # Calculate RSI
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.append(rsi)

    # High volatility measures
    if volatility_regime > 0.7:
        new_features.append(np.std(closing_prices))  # Volatility measure in high volatility
        
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Reward logic for various regimes
    if abs(trend_strength) > 0.3:  # TREND regimes
        if trend_strength > 0:  # Uptrend
            reward += 20.0  # Base reward for trend-following
        else:  # Downtrend
            reward -= 10.0  # Cautious reward in downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if regime_vector[3] > 0:  # If mean-reversion opportunity
            reward += 10.0  # Positive for mean-reversion
        else:
            reward -= 5.0  # Penalize for chasing breakouts

    # Adjust for volatility
    if regime_vector[1] > 0.7:  # HIGH_VOL regimes
        reward -= 15.0  # Scale down all rewards in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within specified bounds