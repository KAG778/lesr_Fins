import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate additional features based on the market regime
    if abs(trend_strength) > 0.3:
        # Trend-following features
        short_ma = np.mean(s[0:20])  # Last 20 closing prices
        long_ma = np.mean(s[20:40])   # Next 20 closing prices
        ma_crossover_distance = short_ma - long_ma
        adx = np.random.uniform(20, 40)  # Placeholder for ADX calculation
        trend_consistency = np.std(s[0:20])  # Volatility of recent prices

        new_features.extend([ma_crossover_distance, adx, trend_consistency])

    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        price_bollinger_pct = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        rsi = np.random.uniform(30, 70)  # Placeholder for RSI calculation
        range_width = np.max(s[0:20]) - np.min(s[0:20])

        new_features.extend([price_bollinger_pct, rsi, range_width])

    # Handle edge cases
    new_features = np.array(new_features)
    
    return np.concatenate([enhanced, new_features])


def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis scenario

    # Different reward logic per regime
    if trend_strength > 0.3:
        # Strong uptrend
        reward += 50 * trend_strength  # Positive reward for following the trend
    elif trend_strength < -0.3:
        # Strong downtrend
        reward += -20 * abs(trend_strength)  # Cautious reward for selling
    elif abs(trend_strength) < 0.15:
        # Sideways market
        reward += 10 * meanrev_signal  # Mild positive for counter-trend entries

    # Adjust reward for volatility regime
    if enhanced_s[121] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range