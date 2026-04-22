import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and the regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Compute features based on regime
    if abs(trend_strength) > 0.3:
        # Trend-following features
        # Example of moving averages
        closing_prices = s[0:20]
        short_ma = np.mean(closing_prices[-5:])  # 5-day MA
        long_ma = np.mean(closing_prices[-15:])  # 15-day MA
        ma_crossover_distance = short_ma - long_ma
        
        # ADX (Average Directional Index)
        # Placeholder for ADX calculation
        adx = np.random.uniform(0, 100)  # replace with actual ADX calculation

        # Trend consistency could be the rate of change of closing prices
        trend_consistency = np.mean(np.diff(closing_prices[-5:]))  # last 5 days

        new_features.extend([ma_crossover_distance, adx, trend_consistency])
        
    else:
        # Mean-reversion features
        # Bollinger Bands
        moving_avg = np.mean(s[0:20])
        std_dev = np.std(s[0:20])
        upper_band = moving_avg + 2 * std_dev
        lower_band = moving_avg - 2 * std_dev
        bollinger_percent_b = (s[0] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
        
        # RSI (Relative Strength Index)
        # Placeholder for RSI calculation
        rsi = np.random.uniform(0, 100)  # replace with actual RSI calculation
        
        new_features.extend([bollinger_percent_b, rsi, std_dev])
    
    # High Volatility features
    if volatility_regime > 0.7:
        # Average True Range (ATR) or similar volatility breakout signals
        atr = np.random.uniform(0, 5)  # replace with actual ATR calculation
        new_features.append(atr)
    
    # Crisis features
    if crisis_signal > 0.5:
        # Defensive indicators like max drawdown
        max_drawdown = np.random.uniform(0, 0.3)  # placeholder for max drawdown calculation
        new_features.append(max_drawdown)
    
    # Combine all features into the enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on regime
    if trend_strength > 0.3:  # Strong uptrend
        if enhanced_s[0] < enhanced_s[1]:  # If today's close is less than yesterday's close
            reward += 25  # Positive reward for following trend
        else:
            reward -= 10  # Cautious reward for trend-following in down moment

    elif trend_strength < -0.3:  # Strong downtrend
        if enhanced_s[0] > enhanced_s[1]:  # If today's close is greater than yesterday's close
            reward -= 20  # Negative reward for going against the trend
        else:
            reward += 10  # Cautious reward for downtrend alignment

    else:  # Sideways market
        if enhanced_s[126] < 0:  # If Bollinger %B indicates mean-reversion opportunity
            reward += 15  # Positive reward for counter-trend entries
    
    # Implementing volatility-adaptive rewards
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude
    
    return reward