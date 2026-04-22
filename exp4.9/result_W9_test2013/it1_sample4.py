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

    # Calculate historical volatility (standard deviation of closing prices)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # last 5 days close
        long_ma = np.mean(closing_prices[-20:])  # last 20 days close
        ma_crossover_distance = short_ma - long_ma

        # ADX (Average Directional Index) calculation (simplified)
        adx = np.mean(np.abs(np.diff(closing_prices)))  # Simplified ADX calculation

        # Normalize the distance by historical volatility
        normalized_ma_distance = ma_crossover_distance / (historical_volatility + 1e-5)  # Avoid division by zero

        new_features = [normalized_ma_distance, adx]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)
        bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower + 1e-5)

        # RSI (Relative Strength Index) calculation
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)

        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-5)))  # Simplified RSI

        new_features = [bollinger_percent_b, rsi]

    # Ensure new features are in a NumPy array
    new_features = np.array(new_features)

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Adjust reward based on market regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Strong positive reward for momentum-aligned entries
            if regime_vector[2] > 0:  # Momentum aligned
                reward += 20.0  # Extra reward for aligned momentum
        else:  # Downtrend
            reward -= 20.0  # Cautious negative reward for downtrend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion signal
            reward += 30.0  # Positive for following mean-reversion opportunities
        else:  # If there's no mean-reversion opportunity
            reward -= 10.0  # Mild penalty for chasing breakouts

    # Adjust for high volatility
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards
        reward -= 10.0  # Additional penalty for aggressive positions

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]