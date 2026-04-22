import numpy as np

def calculate_historical_volatility(prices):
    """Calculate historical volatility as the standard deviation of log returns."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns)

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(s[0:20])
    
    # Regime-conditioned feature extraction
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(s[0:10])  # Short moving average (10 days)
        long_ma = np.mean(s[10:20])   # Long moving average (20 days)
        ma_crossover_distance = short_ma - long_ma
        
        # Add ADX or similar trend strength indicator placeholder
        adx = np.random.uniform(20, 40)  # Placeholder for ADX calculation

        new_features.extend([ma_crossover_distance, adx, historical_volatility])

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        bollinger_mid = np.mean(s[0:20])  # Middle of Bollinger bands
        bollinger_std = np.std(s[0:20])    # Standard deviation
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        
        price_bollinger_pct = (s[0] - bollinger_lower) / (bollinger_upper - bollinger_lower)  # %B
        new_features.append(price_bollinger_pct)

    # Handle edge cases for new features
    new_features = np.nan_to_num(np.array(new_features), nan=0.0)  # Replace NaNs with 0

    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative for any entry during crisis

    # Reward logic based on the regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if momentum_signal > 0:
            reward += 30 * trend_strength  # Strong positive reward for momentum-aligned entries
        else:
            reward -= 15  # Penalize for counter-trend moves

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if meanrev_signal < 0:  # Potential for mean reversion
            reward += 20  # Reward for mean reversion signals
        else:
            reward -= 10  # Penalize for chasing breakouts

    # Adjust reward based on volatility regime
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward stays within range