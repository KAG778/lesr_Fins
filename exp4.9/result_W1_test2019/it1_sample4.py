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
    
    # Use a simple moving average as a feature
    moving_average_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
    moving_average_long = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
    
    # Avoid division by zero for MA ratio
    ma_ratio = (moving_average_short - moving_average_long) / (moving_average_long if moving_average_long != 0 else 1)

    # Determine features based on regimes
    if abs(trend_strength) > 0.3:  # TREND regime
        # Adding features that are useful in trending markets
        adx = np.random.uniform(0, 1)  # Placeholder for ADX calculation
        new_features.extend([ma_ratio, adx, np.mean(closing_prices[-3:])])  # MA ratio, ADX, recent price
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Features useful in sideways markets
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + (2 * bollinger_std)
        bollinger_lower = bollinger_mid - (2 * bollinger_std)
        bollinger_pct_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower if (bollinger_upper - bollinger_lower) != 0 else 1)
        
        rsi = np.random.uniform(0, 100)  # Placeholder for RSI calculation
        new_features.extend([bollinger_pct_b, rsi, np.std(closing_prices)])  # %B, RSI, price volatility
    else:  # CRISIS or other regimes
        # Use crisis features if applicable
        new_features.extend([np.std(closing_prices[-5:]), np.mean(closing_prices[-10:])])  # Adding volatility and mean of last 10 prices

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Determine reward logic based on regimes
    if trend_strength > 0.3:  # TREND UP
        if enhanced_s[125] > 0:  # Assuming new feature at index 125 is momentum
            reward += 50  # Positive reward for aligned momentum
        else:
            reward -= 25  # Penalize misaligned momentum
    elif trend_strength < -0.3:  # TREND DOWN
        reward -= 50  # Strong negative for long positions
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        if enhanced_s[125] < 0:  # Assuming new feature at index 125 is %B
            reward += 30  # Positive for counter-trend opportunity
        else:
            reward -= 15  # Penalize for chasing breakouts
    
    # Adjust for volatility
    if volatility_regime > 0.7:  # HIGH VOL
        reward *= 0.5  # Scale down all rewards

    # Ensure reward is within the specified range
    reward = np.clip(reward, -100, 100)

    return reward