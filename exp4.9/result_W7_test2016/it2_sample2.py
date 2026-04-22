import numpy as np

def calculate_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def compute_bollinger_bands(prices, period):
    """Compute Bollinger Bands."""
    mid = np.mean(prices[-period:])
    std_dev = np.std(prices[-period:])
    upper_band = mid + 2 * std_dev
    lower_band = mid - 2 * std_dev
    return mid, upper_band, lower_band

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    historical_volatility = calculate_historical_volatility(s[0:20])  # Calculate historical volatility

    # Trend-following features for TREND regimes (|trend_strength| > 0.3)
    if abs(trend_strength) > 0.3:
        short_ma = np.mean(s[0:10])  # Short moving average (10 days)
        long_ma = np.mean(s[10:20])   # Long moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.sum(np.diff(s[0:20]) > 0) / 19  # Trend consistency measure

        new_features.extend([ma_crossover_distance, trend_consistency, historical_volatility])

    # Mean-reversion features for SIDEWAYS regimes (|trend_strength| < 0.15)
    elif abs(trend_strength) < 0.15:
        bollinger_mid, bollinger_upper, bollinger_lower = compute_bollinger_bands(s[0:20], 20)
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        new_features.append(bollinger_percent_b)

    # Append new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis scenario

    historical_volatility = calculate_historical_volatility(enhanced_s[:20])  # Calculate historical volatility
    
    # Define volatility-adaptive thresholds for reward scaling
    vol_threshold = historical_volatility * 1.5  # Scale based on historical volatility

    # Reward logic based on regime
    if trend_strength > 0.3:  # TREND UP
        reward += 50 * trend_strength  # Positive reward for aligning with the trend
    elif trend_strength < -0.3:  # TREND DOWN
        reward -= 20 * abs(trend_strength)  # Cautious reward for selling
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS market
        meanrev_signal = regime_vector[3]
        if meanrev_signal < 0:  # Potential buy opportunity in a mean-reversion context
            reward += 10  # Reward for counter-trend buy
        elif meanrev_signal > 0:  # Potential sell opportunity in a mean-reversion context
            reward -= 10  # Penalize for chasing a breakout

    # High volatility regime adjustment
    if regime_vector[1] > 0.7:  # HIGH VOLATILITY
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)  # Ensure reward stays within range