import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # Base 125 dims
    new_features = []

    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero

    # Define thresholds based on historical volatility
    trend_threshold = 0.3 * historical_volatility
    sideways_threshold = 0.15 * historical_volatility

    if abs(trend_strength) > trend_threshold:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
        ma_crossover_distance = short_ma - long_ma  # Difference between MAs

        adx = np.mean(np.abs(np.diff(closing_prices)))  # Simplified ADX
        trend_consistency = np.sign(closing_prices[-1] - closing_prices[0])  # Consistency measure

        new_features = [
            ma_crossover_distance,
            adx,
            trend_consistency
        ]

    elif abs(trend_strength) < sideways_threshold:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)
        bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower + 1e-5)  # Avoid division by zero

        rsi = 100 - (100 / (1 + np.mean(closing_prices[-5:]) / (np.std(closing_prices[-5:]) + 1e-5)))  # Simplified RSI
        range_width = np.max(closing_prices) - np.min(closing_prices)  # Range width

        new_features = [
            bollinger_percent_b,
            rsi,
            range_width
        ]

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative reward for any entries in crisis

    # Define thresholds for regimes
    trend_threshold = 0.3
    sideways_threshold = 0.15

    if abs(trend_strength) > trend_threshold:  # TREND regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 50.0  # Strong positive reward
        elif trend_strength < 0:  # Downtrend
            reward -= 20.0  # Cautious negative reward

    elif abs(trend_strength) < sideways_threshold:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 30.0  # Positive reward for mean reversion
        else:
            reward -= 10.0  # Negative for chasing breakouts

    # High volatility impact
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]