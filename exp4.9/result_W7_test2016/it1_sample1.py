import numpy as np

def calculate_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def compute_moving_average(prices, period):
    """Compute moving average for the given period."""
    return np.mean(prices[-period:]) if len(prices) >= period else np.nan

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(s[0:20])

    if abs(trend_strength) > 0.3:
        # TREND regime
        short_ma = compute_moving_average(s[0:20], 5)  # Short MA
        long_ma = compute_moving_average(s[0:20], 20)  # Long MA
        ma_crossover_distance = short_ma - long_ma
        trend_consistency = np.sum(np.diff(s[0:20]) > 0) / 19  # Trend consistency
        
        new_features.extend([ma_crossover_distance, trend_consistency, historical_volatility])
    
    elif abs(trend_strength) < 0.15:
        # SIDEWAYS regime
        bollinger_mid = np.mean(s[0:20])
        bollinger_std = np.std(s[0:20])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        
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

    # Reward logic adapts based on regime
    if trend_strength > 0.3:  # TREND UP
        reward += 50 * trend_strength  # Strong positive reward for trend alignment
    elif trend_strength < -0.3:  # TREND DOWN
        reward -= 20 * abs(trend_strength)  # Caution for selling in downtrend
    elif abs(trend_strength) < 0.15:  # SIDEWAYS
        meanrev_signal = regime_vector[3]
        if meanrev_signal < 0:  # Potential buy opportunity in a mean-reversion context
            reward += 10  # Reward for counter-trend buy
        elif meanrev_signal > 0:  # Potential sell opportunity in a mean-reversion context
            reward -= 10  # Penalize for chasing a breakout

    # Adjust reward for high volatility regime
    if regime_vector[1] > 0.7:  # HIGH VOLATILITY
        reward *= 0.5  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward stays within range