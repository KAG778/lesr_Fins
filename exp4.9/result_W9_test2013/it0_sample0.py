import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state by concatenating raw state with regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate recent returns
    recent_returns = (s[0] - s[19]) / s[19] if s[19] != 0 else 0.0

    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Simple Moving Average (SMA) Crossover Distance
        short_ma = np.mean(s[0:5])  # last 5 days close
        long_ma = np.mean(s[15:20])  # last 20 days close
        ma_crossover_distance = (short_ma - long_ma) / (long_ma if long_ma != 0 else 1)

        # Average Directional Index (ADX) (mock value)
        adx = np.random.uniform(20, 50)  # Placeholder for actual calculation

        # Trend consistency (simple measure)
        trend_consistency = np.sign(trend_strength) * (np.abs(trend_strength) ** 2)
        
        new_features = [
            ma_crossover_distance,
            adx,
            trend_consistency,
            recent_returns
        ]
    
    else:
        # Mean-reversion features
        # Bollinger %B
        moving_avg = np.mean(s[0:20])
        std_dev = np.std(s[0:20])
        bollinger_percentage_b = (s[0] - (moving_avg - 2 * std_dev)) / (4 * std_dev) if std_dev != 0 else 0.0

        # RSI (mock value)
        rsi = np.random.uniform(30, 70)  # Placeholder for actual calculation

        # Range width
        price_range = s[40:59].max() - s[60:79].min()  # High - Low
        range_width = price_range / (s[60:79].mean() if s[60:79].mean() != 0 else 1)

        new_features = [
            bollinger_percentage_b,
            rsi,
            range_width,
            recent_returns
        ]

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Adjust reward based on market regime
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 10.0  # Positive reward for following trend
        elif trend_strength < 0:  # Downtrend
            reward -= 5.0  # Cautious reward
    else:  # Sideways regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 5.0  # Positive for mean-reversion signal

    # High volatility impact
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 2.0  # Negative for aggressive entries

    return reward