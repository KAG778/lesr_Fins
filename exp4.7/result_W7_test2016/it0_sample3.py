import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, span):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros_like(prices)
    ema[:span] = np.nan  # Initial values will be NaN for the first 'span' values
    ema[span-1] = np.mean(prices[:span])  # First EMA value is the SMA
    for i in range(span, len(prices)):
        ema[i] = (prices[i] * (2 / (span + 1))) + (ema[i-1] * (1 - (2 / (span + 1))))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(window) / window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window) / window, mode='valid')
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window-1, np.nan), rsi))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    return macd

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window) / window, mode='valid')
    return np.concatenate((np.full(window-1, np.nan), atr))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:120]
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Combine all into enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-20:],  # Last 20 values of SMA 5
        sma_10[-20:], # Last 20 values of SMA 10
        sma_20[-20:], # Last 20 values of SMA 20
        ema_5[-20:],  # Last 20 values of EMA 5
        ema_10[-20:], # Last 20 values of EMA 10
        rsi[-20:],    # Last 20 values of RSI
        macd[-20:],   # Last 20 values of MACD
        atr[-20:]     # Last 20 values of ATR
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    
    reward = 0
    
    # Reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Additional checks for trend and momentum
    sma_5 = enhanced_s[120:140]
    sma_10 = enhanced_s[140:160]

    if sma_5[-1] > sma_10[-1]:  # Uptrend
        reward += 30
    elif sma_5[-1] < sma_10[-1]:  # Downtrend
        reward -= 30

    # Risk assessment
    if np.abs(recent_return) > threshold:
        reward -= 20  # High risk
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]