import numpy as np

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return np.convolve(data, np.ones(window), 'valid') / window

def calculate_ema(data, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Start with the first value
    for i in range(1, len(data)):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(data, window):
    """Calculate Relative Strength Index."""
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window - 1, np.nan), rsi))  # Return NaN for unavailable values

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volume = s[80:99]
    adj_closing_prices = s[100:119]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Extend the state with the new features, filling NaNs with zeros or other appropriate values
    enhanced_s = np.concatenate((
        s,
        np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan),
        np.pad(ema_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(ema_10, (9, 0), 'constant', constant_values=np.nan),
        np.pad(rsi_14, (13, 0), 'constant', constant_values=np.nan)
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Daily volatility in percentage
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol
    
    # Initialize reward
    reward = 0
    
    # Determine reward based on recent return and thresholds
    if recent_return > threshold:
        reward += 50  # Good upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Bad downward momentum
    
    # Example of using RSI for additional reward modification
    rsi = enhanced_s[119]  # Get the last RSI value (after enhancements)
    if rsi < 30:
        reward += 20  # Potential buy signal
    elif rsi > 70:
        reward -= 20  # Potential sell signal
    
    return np.clip(reward, -100, 100)  # Ensure reward stays within the specified range