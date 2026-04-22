import numpy as np

def calculate_rolling_mean(data, window):
    """ Helper function to compute rolling mean """
    return np.convolve(data, np.ones(window), 'valid') / window

def calculate_rolling_std(data, window):
    """ Helper function to compute rolling standard deviation """
    return np.sqrt(np.convolve((data - np.mean(data))**2, np.ones(window), 'valid') / window)

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Extract closing prices for feature calculations
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate daily returns for volatility and momentum features
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]

    if abs(trend_strength) > 0.3:
        # Trend-following features
        short_sma = np.mean(closing_prices[-5:])  # Last 5 days
        long_sma = np.mean(closing_prices[-20:])   # Last 20 days
        sma_crossover_distance = short_sma - long_sma
        
        adx = np.std(daily_returns[-14:]) * 100  # Placeholder for ADX
        
        new_features.extend([sma_crossover_distance, adx])

    elif abs(trend_strength) < 0.15:
        # Mean-reversion features
        std_dev = np.std(closing_prices)
        moving_average = np.mean(closing_prices)
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_percent_b = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
        
        gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
        losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        new_features.extend([bollinger_percent_b, rsi])

    # Volatility feature: Average True Range (ATR)
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
                  )  
    new_features.append(atr)

    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis

    # Reward logic based on regime
    if abs(trend_strength) > 0.3:  # Strong trend
        if trend_strength > 0 and regime_vector[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Positive for trend-following
        elif trend_strength < 0:  # Downtrend
            reward -= 10  # Cautious for downtrend

    elif abs(trend_strength) < 0.15:  # Sideways
        if regime_vector[3] > 0:  # Mean-reversion opportunity
            reward += 15  # Positive for mean-reversion
        
    # Adjust for high volatility
    if volatility_regime > 0.7:  # High volatility
        reward -= 15  # Penalize aggressive entries

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]