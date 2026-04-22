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
    
    if abs(trend_strength) > 0.3:  # TREND regimes
        # Calculate trend-following features
        short_ma = np.mean(closing_prices[-5:])  # 5-day MA
        long_ma = np.mean(closing_prices[-20:])  # 20-day MA
        new_features.append(short_ma - long_ma)  # MA Crossover Distance
        
        # Mock ADX calculation
        adx = np.random.uniform(20, 50)
        new_features.append(adx)

        # Trend consistency
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / len(closing_prices[:-1])
        new_features.append(trend_consistency)

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        # Calculate mean-reversion features
        std_dev = np.std(closing_prices)
        bollinger_upper = np.mean(closing_prices) + 2 * std_dev
        bollinger_lower = np.mean(closing_prices) - 2 * std_dev
        percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) if (bollinger_upper - bollinger_lower) > 0 else 0
        new_features.append(percent_b)

        # RSI calculation
        rsi = np.random.uniform(30, 70)  # Placeholder for actual RSI
        new_features.append(rsi)

        # Price range
        price_range = np.max(closing_prices) - np.min(closing_prices)
        new_features.append(price_range)

    # Ensure to return concatenated enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Regime-specific reward logic
    if abs(trend_strength) > 0.3:  # TREND regimes
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and momentum aligned
            reward += 40.0  # Strong positive reward
        else:
            reward += -20.0  # Cautious negative for counter-trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 20.0  # Mild positive for counter-trend entries
        else:
            reward += -10.0  # Penalize breakout chases

    # Adjust for high volatility
    if volatility_regime > 0.7:  # High volatility
        reward *= 0.5  # Scale down rewards in high volatility

    # Clip the reward to ensure it is in the range [-100, 100]
    return np.clip(reward, -100, 100)