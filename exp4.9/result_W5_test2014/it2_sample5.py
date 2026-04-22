import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Extract last 20 days of closing prices for calculations
    closing_prices = s[0:20]
    
    # Avoid division by zero
    if len(closing_prices) > 1 and np.std(closing_prices) > 0:
        # Trend-following features for trend regimes
        if abs(trend_strength) > 0.3:
            # Moving Average Crossover Distance
            short_ma = np.mean(closing_prices[-5:])  # 5-day MA
            long_ma = np.mean(closing_prices[-20:])  # 20-day MA
            ma_crossover_distance = short_ma - long_ma
            new_features.append(ma_crossover_distance)

            # Average True Range (ATR) for volatility
            true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                     np.abs(closing_prices[1:] - closing_prices[:-1]))
            atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
            new_features.append(atr)

            # Trend consistency ratio
            trend_consistency = np.sum(np.diff(closing_prices) > 0) / (len(closing_prices) - 1)
            new_features.append(trend_consistency)

        # Mean-reversion features for sideways regimes
        elif abs(trend_strength) < 0.15:
            # Bollinger %B
            std_dev = np.std(closing_prices)
            bollinger_upper = np.mean(closing_prices) + 2 * std_dev
            bollinger_lower = np.mean(closing_prices) - 2 * std_dev
            percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) > 0 else 0
            new_features.append(percent_b)

            # RSI calculation
            rsi = np.sum(closing_prices[-14:] > np.mean(closing_prices[-14:])) / 14 * 100  # Simplified RSI
            new_features.append(rsi)

            # Price range
            price_range = np.max(closing_prices) - np.min(closing_prices)
            new_features.append(price_range)

    # Add new features to the enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Reward logic based on regimes
    if abs(trend_strength) > 0.3:  # Trend regime
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 50.0  # Strong positive reward
        elif trend_strength < 0:  # Downtrend
            reward += 10.0  # Cautious positive reward
        else:
            reward += -10.0  # Penalize if trend strength is neutral
        
    elif abs(trend_strength) < 0.15:  # Sideways regime
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Positive for counter-trend entries
        else:
            reward += -10.0  # Penalize breakout chases

    # Volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds