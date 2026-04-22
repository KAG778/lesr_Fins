import numpy as np

def revise_state(s, regime_vector):
    # Extract regime information
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Base state concatenation
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Extract last 20 days of closing prices
    closing_prices = s[0:20]
    if len(closing_prices) > 1 and np.std(closing_prices) > 0:
        if abs(trend_strength) > 0.3:  # Trend regime
            # Trend-following features
            short_ma = np.mean(closing_prices[-5:])  # Short-term MA
            long_ma = np.mean(closing_prices[-20:])  # Long-term MA
            new_features.append(short_ma - long_ma)  # MA Crossover
            
            true_ranges = np.maximum(s[40:60][1:] - s[60:80][1:], 
                                     np.maximum(np.abs(s[40:60][1:] - closing_prices[:-1]), 
                                                np.abs(s[60:80][1:] - closing_prices[:-1])))
            atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
            new_features.append(atr)  # Average True Range
            
            trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices[:-1])
            new_features.append(trend_consistency)  # Trend consistency ratio

        elif abs(trend_strength) < 0.15:  # Sideways regime
            # Mean-reversion features
            std_dev = np.std(closing_prices)
            bollinger_upper = np.mean(closing_prices) + 2 * std_dev
            bollinger_lower = np.mean(closing_prices) - 2 * std_dev
            percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) > 0 else 0
            new_features.append(percent_b)  # Bollinger %B

            rsi = (np.sum(closing_prices[-14:] > np.mean(closing_prices[-14:])) / 14) * 100  # Simplified RSI
            new_features.append(rsi)  # Relative Strength Index
            
            price_range = np.max(closing_prices) - np.min(closing_prices)
            new_features.append(price_range)  # Price range

    # Concatenate all features
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    # Extract regime_vector from enhanced_state[120:125]
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative reward for any entries during a crisis

    # Reward logic based on regimes
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 50.0  # Strong positive reward
        elif trend_strength < 0:  # Downtrend
            reward += 10.0  # Cautious positive reward
    elif abs(trend_strength) < 0.15:  # Sideways regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Positive for mean-reversion
        else:
            reward += -10.0  # Penalize breakout chases

    # Volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Scale down rewards in high volatility

    # Clip the reward to ensure it is in the range [-100, 100]
    return np.clip(reward, -100, 100)