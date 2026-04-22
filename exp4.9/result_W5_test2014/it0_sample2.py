import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the raw state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Initialize new features list
    new_features = []

    # Extract last 20 days of closing prices for calculations
    closing_prices = s[0:20]
    
    # Avoid division by zero
    if len(closing_prices) > 1 and np.std(closing_prices) > 0:
        # Trend-following features
        if abs(trend_strength) > 0.3:
            # Moving Average Crossover Distance (short MA - long MA)
            short_ma = np.mean(closing_prices[-5:])  # 5-day MA
            long_ma = np.mean(closing_prices[-20:])  # 20-day MA
            new_features.append(short_ma - long_ma)
            
            # Average Directional Index (ADX)
            # Here we mock the ADX calculation as an example
            adx = np.random.uniform(20, 50)  # Placeholder for actual ADX calculation
            new_features.append(adx)

            # Trend consistency (simple measure)
            trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices[:-1])  # Proportion of positive daily changes
            new_features.append(trend_consistency)

        # Mean-reversion features
        else:
            # Bollinger %B
            std_dev = np.std(closing_prices)
            bollinger_upper = np.mean(closing_prices) + 2 * std_dev
            bollinger_lower = np.mean(closing_prices) - 2 * std_dev
            percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper - bollinger_lower > 0 else 0
            new_features.append(percent_b)

            # RSI calculation (mocking as well)
            rsi = np.random.uniform(30, 70)  # Placeholder for actual RSI calculation
            new_features.append(rsi)

            # Range width
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
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on regimes
    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 50.0  # Strong positive reward
        elif trend_strength < 0:  # Downtrend
            reward += 10.0  # Cautious positive reward
    else:  # Sideways
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Mild positive for counter-trend entries

    # High volatility penalty
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 20.0  # Negative for aggressive entries

    return reward