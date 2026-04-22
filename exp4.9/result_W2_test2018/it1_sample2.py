import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base enhanced state with raw state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions
    new_features = []
    
    closing_prices = s[:20]
    historical_volatility = np.std(closing_prices)  # 20-day historical volatility

    # Feature extraction based on trend
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        long_ma = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.mean(np.diff(closing_prices) > 0)  # Share of positive returns

        new_features = [ma_crossover_distance, trend_consistency, historical_volatility]
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + (bollinger_std * 2)
        lower_bollinger = bollinger_mid - (bollinger_std * 2)
        bollinger_pct_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        rsi = np.mean(np.diff(closing_prices) > 0)  # Simplified RSI calculation

        new_features = [bollinger_pct_b, rsi, historical_volatility]

    if volatility_regime > 0.7:  # HIGH_VOL regime
        new_features.append(historical_volatility)  # Include historical volatility as a feature

    return np.concatenate([enhanced, new_features])


def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis regime

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if regime_vector[2] > 0:  # Strong upward momentum
            reward += 50.0  # Positive reward for aligning with trend
        else:  # Downward momentum
            reward -= 30.0  # Penalize for counter-trend action

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] < 0:  # Price at lower Bollinger
            reward += 20.0  # Reward for mean-reversion opportunities
        elif regime_vector[3] > 0:  # Price at upper Bollinger
            reward -= 20.0  # Penalize for breakout attempts

    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward -= 10.0  # Penalize for aggressive entries when volatility is high

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]