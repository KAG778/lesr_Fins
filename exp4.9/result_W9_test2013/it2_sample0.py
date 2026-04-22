import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    enhanced = np.concatenate([s, regime_vector])  # Base 125 dims
    new_features = []

    # Calculate historical volatility (standard deviation of closing prices)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero

    # Define adaptive thresholds based on historical volatility
    trend_threshold = 0.3 * historical_volatility
    sideways_threshold = 0.15 * historical_volatility

    # Determine market regime and compute features accordingly
    if abs(trend_strength) > trend_threshold:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
        ma_crossover_distance = short_ma - long_ma

        # Average Directional Index (ADX) calculation (simplified)
        adx = np.mean(np.abs(np.diff(closing_prices)))

        # Include a measure of trend consistency
        trend_consistency = np.sign(trend_strength)

        new_features = [
            ma_crossover_distance,
            adx,
            trend_consistency
        ]

    elif abs(trend_strength) < sideways_threshold:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)
        bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower + 1e-5)

        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-5))) if avg_loss != 0 else 100

        range_width = np.max(closing_prices) - np.min(closing_prices)

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
        return -100.0  # Strong negative in crisis

    # Define adaptive thresholds for regimes
    trend_threshold = 0.3
    sideways_threshold = 0.15

    if abs(trend_strength) > trend_threshold:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Strong positive reward for following trend
            if regime_vector[2] > 0:  # Momentum aligned
                reward += 20.0  # Extra positive for momentum alignment
        else:  # Downtrend
            reward -= 20.0  # Cautious negative reward

    elif abs(trend_strength) < sideways_threshold:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 30.0  # Positive for mean-reversion signals
        else:  # Penalize chasing breakouts
            reward -= 10.0  # Mild negative for chasing breakouts

    # High volatility impact
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards
        reward -= 10.0  # Additional penalty for aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]