import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate the closing prices
    closing_prices = s[0:20]

    if abs(trend_strength) > 0.3:
        # Trend-following features
        moving_average_short = np.mean(closing_prices[-5:])  # Short MA (last 5 days)
        moving_average_long = np.mean(closing_prices[-20:])  # Long MA (last 20 days)
        ma_ratio = (moving_average_short - moving_average_long) / (moving_average_long if moving_average_long != 0 else 1)
        
        adx = np.random.uniform(0, 1)  # Placeholder for ADX calculation
        new_features.extend([ma_ratio, adx, closing_prices[-1]])  # MA ratio, ADX, last closing price
    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        bollinger_upper = bollinger_mid + (2 * bollinger_std)
        bollinger_lower = bollinger_mid - (2 * bollinger_std)
        bollinger_pct_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower if (bollinger_upper - bollinger_lower) != 0 else 1)

        rsi = np.random.uniform(0, 100)  # Placeholder for RSI calculation
        new_features.extend([bollinger_pct_b, rsi, np.std(closing_prices)])  # %B, RSI, price volatility

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        # Strong negative reward for entries in crisis
        if enhanced_s[125] > 0:  # Assuming momentum signal is at index 125
            return -100.0  # Very strong penalty for buying in crisis
        else:
            return 20.0  # Mild positive for selling/holding in crisis
    
    # Different reward logic per regime
    if trend_strength > 0.3:  # Trend Up
        if enhanced_s[125] > 0:  # Assuming new feature at index 125 is momentum
            reward += 50  # Strong positive reward for aligned momentum
        else:
            reward -= 30  # Strong penalty for misaligned momentum
    elif trend_strength < -0.3:  # Trend Down
        reward -= 50  # Strong penalty for any long positions
    elif abs(trend_strength) < 0.15:  # Sideways Market
        if enhanced_s[125] < 0:  # Assuming new feature at index 125 is %B
            reward += 30  # Positive for counter-trend opportunity
        else:
            reward -= 20  # Penalty for chasing breakouts
    
    # Adjust for volatility
    if enhanced_s[121] > 0.7:  # Assuming volatility regime at index 121
        reward -= 20  # Penalty for aggressive entries in high volatility

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)