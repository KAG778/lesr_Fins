import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate moving averages for trend-following features
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    
    # Calculate simple moving averages
    short_ma = np.mean(closing_prices[-5:])  # Last 5 days
    long_ma = np.mean(closing_prices[-20:])  # Last 20 days
    
    # Calculate Average True Range (ATR) for volatility
    high_prices = s[40:60]
    low_prices = s[60:80]
    true_ranges = np.maximum(high_prices[1:], closing_prices[:-1]) - np.minimum(low_prices[1:], closing_prices[:-1])
    atr = np.mean(true_ranges[-14:])  # Last 14 days for ATR
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        new_features.append(short_ma - long_ma)  # MA Crossover distance
        new_features.append(atr)                  # Average True Range
        new_features.append(np.std(closing_prices))  # Price volatility
    else:
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices[-20:]) + 2 * np.std(closing_prices[-20:])
        lower_bollinger = np.mean(closing_prices[-20:]) - 2 * np.std(closing_prices[-20:])
        new_features.append((closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger))  # Bollinger %B
        new_features.append((closing_prices[-1] - np.mean(closing_prices[-20:])) / np.std(closing_prices[-20:]))  # Z-score
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width
        
    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic based on different regimes
    if trend_strength > 0.3:
        # Strong uptrend
        reward += 50  # Positive reward for aligning with the trend
        if regime_vector[2] == 1:  # Strong upward momentum
            reward += 20  # Boost reward due to momentum
        elif regime_vector[2] == -1:  # Strong downward momentum
            reward -= 10  # Cautious on downward momentum
    elif trend_strength < -0.3:
        # Strong downtrend
        reward -= 20  # Negative reward for buying in downtrend
    else:
        # Sideways market
        reward += 10  # Mild positive for counter-trend opportunities
        if regime_vector[3] == -1:  # Price at lower Bollinger
            reward += 20  # Good opportunity to bounce
        
    # High volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 10  # Penalize aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]