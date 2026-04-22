import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and append the regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate historical volatility (standard deviation of closing prices)
    closing_prices = s[0:20]
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.01  # Avoid division by zero

    if abs(trend_strength) > 0.3:  # Trend regimes
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # 5-day MA
        moving_average_long = np.mean(closing_prices[-20:])  # 20-day MA
        ma_crossover_distance = moving_average_short - moving_average_long

        # Average Directional Index (ADX) simplification
        adx = np.mean(np.abs(np.diff(closing_prices)))

        # Normalize the recent price to historical volatility
        normalized_recent_price = (closing_prices[-1] - moving_average_short) / (historical_volatility + 1e-5)

        new_features = [ma_crossover_distance, adx, normalized_recent_price]

    elif abs(trend_strength) < 0.15:  # Sideways regimes
        # Mean-reversion features
        bollinger_upper = np.mean(closing_prices) + 2 * np.std(closing_prices)
        bollinger_lower = np.mean(closing_prices) - 2 * np.std(closing_prices)
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower + 1e-5)  # Avoid division by zero

        # RSI calculation
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:])  
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-5))) if avg_loss != 0 else 100

        new_features = [bollinger_percent_b, rsi, np.max(closing_prices) - np.min(closing_prices)]

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Positive reward
            if regime_vector[2] > 0:  # Momentum aligned
                reward += 20.0  # Extra positive for momentum
        else:  # Downtrend
            reward -= 30.0  # Cautious negative reward

    elif abs(trend_strength) < 0.15:  # Sideways regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 30.0  # Positive for mean-reversion signal
        else:  # Avoid chasing breakouts
            reward -= 20.0  # Negative for aggressive entries

    # High volatility impact
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Scale down all rewards
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]