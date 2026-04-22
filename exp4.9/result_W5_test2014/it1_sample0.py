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

    if abs(trend_strength) > 0.3:  # TREND regime
        # Calculate trend-following features
        short_ma = np.mean(closing_prices[-5:])
        long_ma = np.mean(closing_prices[-20:])
        ma_crossover_distance = short_ma - long_ma
        adx = np.random.uniform(20, 50)  # Placeholder for actual ADX calculation
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices[:-1])
        
        new_features.extend([ma_crossover_distance, adx, trend_consistency])
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Calculate mean-reversion features
        std_dev = np.std(closing_prices)
        bollinger_upper = np.mean(closing_prices) + 2 * std_dev
        bollinger_lower = np.mean(closing_prices) - 2 * std_dev
        percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) > 0 else 0
        rsi = np.random.uniform(30, 70)  # Placeholder for actual RSI calculation
        price_range = np.max(closing_prices) - np.min(closing_prices)
        
        new_features.extend([percent_b, rsi, price_range])
    
    # Add volatility features
    if volatility_regime > 0.7:  # HIGH_VOL regime
        # Include features that reflect the high volatility
        new_features.append(np.std(closing_prices))  # Add recent volatility as a feature
        
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Adapting reward structure based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 50.0
        else:  # Downtrend or momentum misaligned
            reward -= 20.0  # Penalize counter-trend entries

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Reward mean-reversion signals
        else:  # Avoid breakout chases
            reward -= 10.0  # Penalize for chasing breakouts
    
    # Adjust reward based on volatility
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds