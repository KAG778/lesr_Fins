import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate the moving averages
    closing_prices = s[0:20]
    moving_average_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
    moving_average_long = np.mean(closing_prices[-20:])  # Long MA (last 20 days)

    # Avoid division by zero for MA ratio
    ma_ratio = (moving_average_short - moving_average_long) / (moving_average_long if moving_average_long != 0 else 1)

    # Feature engineering based on the current regime
    if abs(trend_strength) > 0.3:  # Trend regimes
        adx = np.random.uniform(0, 1)  # Placeholder for ADX calculation
        new_features.extend([ma_ratio, adx, np.mean(closing_prices[-3:])])  # MA ratio, ADX, recent price
    elif abs(trend_strength) < 0.15:  # Sideways regime
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + (2 * bollinger_std)
        bollinger_lower = bollinger_mid - (2 * bollinger_std)
        bollinger_pct_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower if (bollinger_upper - bollinger_lower) != 0 else 1)

        rsi = np.random.uniform(0, 100)  # Placeholder for RSI calculation
        new_features.extend([bollinger_pct_b, rsi, np.std(closing_prices)])  # %B, RSI, price volatility
    else:  # Mean-reversion in a weak trend
        recent_price = closing_prices[-1]
        new_features.extend([ma_ratio, recent_price])  # Include MA ratio and recent price as features

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Different reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        reward += 30  # Base reward for uptrend
        if enhanced_s[125] > 0:  # Assuming new feature at index 125 is momentum
            reward += 20  # Positive reward for aligned momentum
        else:
            reward -= 10  # Penalize for misaligned momentum
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 30  # Strong penalty for going long
    elif abs(trend_strength) < 0.15:  # Sideways market
        if enhanced_s[125] < 0:  # Assuming new feature at index 125 is %B
            reward += 20  # Positive for mean-reverting entries
        else:
            reward -= 10  # Penalize for chasing breakouts

    # Adjust for volatility
    historical_std = np.std(enhanced_s[0:20])  # Calculate historical std for the last 20 periods
    if regime_vector[1] > 0.7:  # High volatility regime
        reward *= 0.5  # Scale down all rewards
        if enhanced_s[125] > 0:  # Assuming new feature at index 125 is momentum
            reward -= 10  # Discourage aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]