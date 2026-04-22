import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    closing_prices = s[0:20]
    moving_average_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
    moving_average_long = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
    
    # Avoid division by zero for MA ratio
    ma_ratio = (moving_average_short - moving_average_long) / (moving_average_long if moving_average_long != 0 else 1)

    # Feature engineering based on the current regime
    if abs(trend_strength) > 0.3:  # TREND regimes
        adx = np.random.uniform(0, 1)  # Placeholder for ADX calculation
        recent_price = closing_prices[-1]
        new_features.extend([ma_ratio, adx, recent_price])  # MA ratio, ADX, recent price
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + (2 * bollinger_std)
        bollinger_lower = bollinger_mid - (2 * bollinger_std)
        bollinger_pct_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower if (bollinger_upper - bollinger_lower) != 0 else 1)
        
        rsi = np.random.uniform(0, 100)  # Placeholder for RSI calculation
        new_features.extend([bollinger_pct_b, rsi, np.std(closing_prices)])  # %B, RSI, price volatility
    else:  # Weak trend or mean-reversion
        recent_price = closing_prices[-1]
        new_features.extend([ma_ratio, recent_price])  # Include MA ratio and recent price as features

    # Adding volatility adaptive features
    historical_std = np.std(closing_prices)
    volatility_adaptive_threshold = historical_std * (1 + volatility_regime)
    new_features.append(volatility_adaptive_threshold)

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
    if trend_strength > 0.3:  # TREND Up
        if enhanced_s[125] > 0:  # Assuming new feature at index 125 is momentum
            reward += 50  # Strong positive reward for aligned momentum
        else:
            reward -= 25  # Penalize misaligned momentum
    elif trend_strength < -0.3:  # TREND Down
        reward -= 50  # Strong penalty for any long positions
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if enhanced_s[125] < 0:  # Assuming new feature at index 125 is %B
            reward += 30  # Positive for counter-trend opportunity
        else:
            reward -= 15  # Penalize for chasing breakouts

    # Adjust for volatility regime
    if regime_vector[1] > 0.7:  # HIGH VOL
        reward *= 0.5  # Scale down all rewards

    # Ensure reward is within the specified range
    return np.clip(reward, -100, 100)