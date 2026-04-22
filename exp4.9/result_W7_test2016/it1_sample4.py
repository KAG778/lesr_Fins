import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate recent closing prices for features
    closing_prices = s[0:20]

    # Use historical volatility for adaptive thresholds
    historical_volatility = np.std(np.log(closing_prices[1:] / closing_prices[:-1]))  # Log returns standard deviation

    if abs(trend_strength) > 0.3:  # TREND regimes
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short moving average (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long moving average (last 20 days)
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / 19  # Trend consistency

        new_features.extend([ma_crossover_distance, trend_consistency])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices[-20:])
        bollinger_std = np.std(closing_prices[-20:])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)

        new_features.append(bollinger_percent_b)

    # Edge case handling for new features
    new_features = np.nan_to_num(np.array(new_features), nan=0.0)  # Replace NaNs with 0

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis scenario

    # Different reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        if momentum_signal > 0:  # Positive momentum
            reward += 30 * trend_strength  # Strongly reward trend alignment
        else:
            reward -= 20 * abs(trend_strength)  # Penalize counter-trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if meanrev_signal < 0:  # Price at lower Bollinger
            reward += 15  # Reward for mean-reversion buy
        elif meanrev_signal > 0:  # Price at upper Bollinger
            reward -= 15  # Penalize for mean-reversion sell

    # Adjust reward based on volatility regime
    if volatility_regime > 0.7:  # HIGH_VOL regimes
        reward *= 0.5  # Scale down the reward

    return np.clip(reward, -100, 100)  # Ensure reward stays within range