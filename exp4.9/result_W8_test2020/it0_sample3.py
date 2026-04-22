import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    # Calculate the 20-day moving average and standard deviation for volatility
    closing_prices = s[0:20]
    moving_average = np.mean(closing_prices)
    moving_std = np.std(closing_prices)
    
    # Feature calculation based on the market regime
    if abs(trend_strength) > 0.3:  # Strong trend regime
        # Trend-following features
        ma_distance = closing_prices[-1] - moving_average  # Distance from moving average
        adx = (closing_prices[-1] - closing_prices[0]) / closing_prices[0]  # Simplistic ADX-like measure
        trend_consistency = np.sign(moving_average - closing_prices[0])  # Direction of trend
        new_features = [ma_distance, adx, trend_consistency]
        
    elif abs(trend_strength) < 0.15:  # Sideways regime
        # Mean-reversion features
        bollinger_percent_b = (closing_prices[-1] - (moving_average - 2 * moving_std)) / (2 * moving_std)  # %B
        rsi = 100 - (100 / (1 + np.mean(closing_prices[-14:]) / np.mean(closing_prices[-14:])) if np.mean(closing_prices[-14:]) != 0 else 1)  # Simplistic RSI
        range_width = np.max(closing_prices) - np.min(closing_prices)  # Price range
        new_features = [bollinger_percent_b, rsi, range_width]

    # Handling volatility regime for feature scaling
    if volatility_regime > 0.7:  # High volatility regime
        atr = np.mean(np.abs(np.diff(closing_prices)))  # Simplistic ATR calculation
        breakout_signal = (closing_prices[-1] - moving_average) / moving_std  # Z-score for breakout
        new_features += [atr, breakout_signal]

    # Crisis regime feature
    if crisis_signal > 0.5:  # Crisis condition
        drawdown_rate = np.min(closing_prices) / np.max(closing_prices) - 1  # Simplistic drawdown rate
        max_consecutive_losses = np.sum(np.array(closing_prices) < moving_average)  # Count of consecutive losses
        new_features += [drawdown_rate, max_consecutive_losses]

    # Ensure new_features is a NumPy array
    new_features = np.array(new_features)
    
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Different reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        if enhanced_s[0] > enhanced_s[1]:  # Assuming last closing price > previous
            reward += 10  # Positive reward for trend-aligned buy
        else:
            reward -= 5  # Cautious reward for counter-trend
    
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 10  # Negative reward for buy signals
    
    elif abs(trend_strength) < 0.15:  # Sideways
        if enhanced_s[0] < enhanced_s[1]:  # Assuming last closing price < previous
            reward += 5  # Mild positive for mean-reversion opportunity
            
    # Adjust reward for high volatility
    volatility_adjustment = (1 - regime_vector[1]) * 100  # Scale based on volatility
    reward *= (volatility_adjustment / 100.0)

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]