import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate moving averages
    closing_prices = s[0:20]
    moving_average_short = np.mean(closing_prices[-5:])  # 5-day MA
    moving_average_long = np.mean(closing_prices[-20:])  # 20-day MA
    
    # Trend-following features
    if abs(trend_strength) > 0.3:
        ma_crossover_distance = moving_average_short - moving_average_long
        adx = np.random.random()  # Placeholder for ADX, should be calculated based on historical data
        trend_consistency = np.random.random()  # Placeholder for consistency measure
        
        new_features.extend([ma_crossover_distance, adx, trend_consistency])
    
    # Mean-reversion features
    elif abs(trend_strength) < 0.15:
        bollinger_mid = np.mean(closing_prices)
        bollinger_std = np.std(closing_prices)
        upper_bollinger = bollinger_mid + 2 * bollinger_std
        lower_bollinger = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger)
        
        rsi = np.random.random()  # Placeholder for RSI calculation
        range_width = np.max(closing_prices) - np.min(closing_prices)
        
        new_features.extend([bollinger_percent_b, rsi, range_width])
    
    # High volatility features
    if volatility_regime > 0.7:
        atr = np.random.random()  # Placeholder for ATR calculation, should be based on historical data
        volatility_breakout_signal = np.random.choice([0, 1])  # Placeholder for breakout signals
        
        new_features.extend([atr, volatility_breakout_signal, np.std(closing_prices)])
    
    # Crisis features
    if crisis_signal > 0.5:
        drawdown_rate = np.random.random()  # Placeholder for drawdown calculation
        max_consecutive_losses = np.random.randint(1, 10)  # Placeholder
        
        new_features.extend([drawdown_rate, max_consecutive_losses, np.mean(s[80:99])])  # Average volume
    
    return np.concatenate([enhanced, new_features])


def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on trend and momentum
    if trend_strength > 0.3:  # Strong upward trend
        if enhanced_s[100] > 0:  # Placeholder for momentum alignment
            reward += 50.0  # Strong positive reward
        else:
            reward += 10.0  # Cautious reward
    
    elif trend_strength < -0.3:  # Strong downward trend
        reward -= 20.0  # Cautious negative reward
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if momentum_signal < 0:  # Potential mean-reversion opportunity
            reward += 20.0  # Mild positive for counter-trend entries
    
    return reward