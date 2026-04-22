import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    close_prices = s[0:20]
    open_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_features = []
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(close_prices, 5)
    sma_10 = calculate_sma(close_prices, 10)
    sma_20 = calculate_sma(close_prices, 20)
    ema_5 = calculate_ema(close_prices, 5)
    ema_10 = calculate_ema(close_prices, 10)
    
    enhanced_features.append(sma_5)
    enhanced_features.append(sma_10)
    enhanced_features.append(sma_20)
    enhanced_features.append(ema_5)
    enhanced_features.append(ema_10)
    enhanced_features.append(close_prices[-1] - sma_5)  # Current price vs 5-day SMA
    enhanced_features.append(close_prices[-1] - sma_10) # Current price vs 10-day SMA
    enhanced_features.append(close_prices[-1] - sma_20) # Current price vs 20-day SMA

    # Momentum Indicators
    rsi_5 = calculate_rsi(close_prices, 5)
    rsi_14 = calculate_rsi(close_prices, 14)
    
    # Simple MACD calculation
    ema_12 = calculate_ema(close_prices, 12)
    ema_26 = calculate_ema(close_prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema([macd_line] + [0]*(len(close_prices)-1), 9)  # assume leading zeros for short history
    macd_histogram = macd_line - signal_line
    
    enhanced_features.append(rsi_5)
    enhanced_features.append(rsi_14)
    enhanced_features.append(macd_line)
    enhanced_features.append(signal_line)
    enhanced_features.append(macd_histogram)

    # Volatility Indicators
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    historical_vol_5 = np.std(daily_returns[-5:]) * 100
    historical_vol_20 = np.std(daily_returns[-20:]) * 100
    atr = np.mean(np.max([high_prices[1:] - low_prices[1:], 
                           np.abs(high_prices[1:] - close_prices[:-1]), 
                           np.abs(low_prices[1:] - close_prices[:-1])], axis=0))
    
    enhanced_features.append(historical_vol_5)
    enhanced_features.append(historical_vol_20)
    enhanced_features.append(atr)

    # Volume-Price Relationship
    obv = np.cumsum(np.where(daily_returns > 0, volumes[1:], 
                             np.where(daily_returns < 0, -volumes[1:], 0)))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
    
    enhanced_features.append(obv[-1])
    enhanced_features.append(volume_ratio)

    # Market Regime Detection
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(close_prices)), close_prices)[0, 1]  # Simplified linear regression R²
    price_position = (close_prices[-1] - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices)) if np.max(close_prices) != np.min(close_prices) else 0
    volume_ratio_regime = (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 0
    
    enhanced_features.append(volatility_ratio)
    enhanced_features.append(trend_strength)
    enhanced_features.append(price_position)
    enhanced_features.append(volume_ratio_regime)

    # Combine original state and new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    close_prices = enhanced_s[0:20]
    daily_returns = np.diff(close_prices) / close_prices[:-1] * 100
    recent_return = daily_returns[-1] if len(daily_returns) > 0 else 0
    
    # Calculate historical volatility
    historical_vol = np.std(daily_returns)
    threshold = 2 * historical_vol  # Relative threshold based on historical volatility
    
    reward = 0

    if position == 0:  # Not holding
        if enhanced_s[-4] > 1:  # Strong uptrend signal (e.g., trend_strength)
            reward += 50
        if recent_return > 0 and recent_return < threshold:  # Positive but low return
            reward += 10
        if recent_return < -threshold:  # Significant loss
            reward -= 50
    else:  # Holding
        if enhanced_s[-4] < 0.5:  # Weak trend signal
            reward -= 50
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        if recent_return > 0:  # Positive return
            reward += 10

    return reward