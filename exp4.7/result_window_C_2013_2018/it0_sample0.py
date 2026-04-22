import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # First 'window' values are NaN
    ema[window-1] = np.mean(prices[:window])
    
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
        
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate((np.full(window-1, np.nan), rsi))  # Prepend NaN values

def calculate_bollinger_bands(prices, window, num_std_dev):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[:window])  # Compute std for initial window
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volume = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20, 2)

    # Create enhanced state
    enhanced_s = np.concatenate((s, sma_5[-20:], sma_10[-20:], ema_5[-20:], rsi_14[-20:], upper_band[-20:], lower_band[-20:]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    rsi = enhanced_s[120:140][-1]  # Latest RSI value
    volume = enhanced_s[80:100][-1]  # Latest trading volume
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Historical volatility
    
    # Calculate relative thresholds
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    # Evaluate reward based on recent return and RSI
    if recent_return > threshold and rsi < 70:
        reward += 50  # Strong upward movement with RSI under 70
    elif recent_return < -threshold:
        reward -= 50  # Significant drop
    
    # Penalize for low volume
    if volume < np.mean(enhanced_s[80:100]) * 0.5:
        reward -= 20  # Low trading volume
    
    return reward