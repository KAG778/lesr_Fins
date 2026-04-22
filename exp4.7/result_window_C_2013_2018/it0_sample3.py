import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # Initialize with NaN for the first 'window' elements
    ema[window-1] = np.mean(prices[:window])  # First EMA is the SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    
    for i in range(window, len(prices)):
        avg_gain = (avg_gain * (window - 1) + gain[i - 1]) / window
        avg_loss = (avg_loss * (window - 1) + loss[i - 1]) / window
        
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate((np.full(window-1, np.nan), rsi))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    return macd

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    
    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    
    # Add new features to the state
    # Padding with NaN for dimensions to match original dimensions
    enhanced_s = np.concatenate([
        s,
        sma_5[-20:],  # Last 20 values
        sma_10[-20:],  # Last 20 values
        rsi[-20:],  # Last 20 values
        macd[-20:]   # Last 20 values
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    rsi = enhanced_s[120:140][-1]  # Last RSI value
    macd = enhanced_s[140:160][-1]  # Last MACD value
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility
    
    reward = 0
    
    # Reward based on RSI
    if rsi < 30:
        reward += 30  # Oversold, potential buy signal
    elif rsi > 70:
        reward -= 30  # Overbought, potential sell signal

    # Reward based on MACD
    if macd > 0:
        reward += 20  # Bullish signal
    elif macd < 0:
        reward -= 20  # Bearish signal

    # Relative threshold for recent return
    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    if recent_return < -threshold:
        reward -= 50  # Penalize for large negative move

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]