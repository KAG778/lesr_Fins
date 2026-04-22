import numpy as np

def compute_moving_average(prices, period):
    """Compute moving average for the given period."""
    return np.mean(prices[-period:]) if len(prices) >= period else np.nan

def compute_bollinger_bands(prices, period):
    """Compute Bollinger Bands."""
    mid = np.mean(prices[-period:])
    std_dev = np.std(prices[-period:])
    upper_band = mid + 2 * std_dev
    lower_band = mid - 2 * std_dev
    return mid, upper_band, lower_band

def calculate_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate historical volatility for scaling rewards
    historical_volatility = calculate_historical_volatility(s[0:20])

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = compute_moving_average(s[0:20], 5)  # Short Moving Average
        long_ma = compute_moving_average(s[0:20], 20)  # Long Moving Average
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.sum(np.diff(s[0:20]) > 0) / 19  # Trend consistency
        
        new_features.extend([ma_crossover_distance, trend_consistency, historical_volatility])
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_mid, bollinger_upper, bollinger_lower = compute_bollinger_bands(s[0:20], 20)
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        
        new_features.extend([bollinger_percent_b, historical_volatility])

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

    # Adjust reward logic based on the regime
    if trend_strength > 0.3:  # TREND regime
        reward += 50 * trend_strength  # Positive reward for following the trend
    elif trend_strength < -0.3:  # STRONG DOWNTREND
        reward -= 20 * abs(trend_strength)  # Caution for selling in downtrend
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        meanrev_signal = regime_vector[3]
        if meanrev_signal < 0:  # Potential buy opportunity in a mean-reversion context
            reward += 15  # Positive for counter-trend buy
        elif meanrev_signal > 0:  # Potential sell opportunity in a mean-reversion context
            reward -= 15  # Penalize for chasing a breakout

    # Adjust reward for high volatility regime
    if regime_vector[1] > 0.7:  # HIGH VOLATILITY
        reward *= 0.5  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward stays within defined range