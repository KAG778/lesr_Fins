import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(high[1:] - low[1:], high[1:] - close[:-1], close[:-1] - low[1:])
    atr = np.zeros_like(close)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume_data = s[80:99]
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Combine features into enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-1:], sma_10[-1:], sma_20[-1:],  # Last SMA values
        ema_5[-1:], ema_10[-1:],  # Last EMA values
        np.array([rsi]),  # Last RSI value
        atr[-1:]  # Last ATR value
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    
    # Determine reward
    reward = 0
    
    # Relative thresholds
    threshold = 2 * historical_vol  # Adaptive threshold based on volatility
    
    if recent_return < -threshold:
        reward -= 50  # Penalty for large negative return
    elif recent_return > threshold:
        reward += 50  # Reward for large positive return
    
    # Additional conditions can be added based on other features if needed
    rsi = enhanced_s[119]  # Assuming RSI is the last feature in enhanced_s
    if rsi < 30:
        reward += 10  # Potentially oversold condition
    elif rsi > 70:
        reward -= 10  # Potentially overbought condition
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]