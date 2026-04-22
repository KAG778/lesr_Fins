import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA) for given prices and window size."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA) for given prices and window size."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window-1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI) for given prices and window size."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate((np.full(window - 1, np.nan), [rsi]))  # Pad with NaN for alignment

def calculate_macd(prices):
    """Calculate MACD for given prices."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12 - ema_26

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR) for given high, low, and close prices."""
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = np.zeros_like(close)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window  # Smoothing
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculating technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Padding indicators to align with the original state dimensions
    sma_5 = np.concatenate((np.full(4, np.nan), sma_5))  # 5-day SMA
    sma_10 = np.concatenate((np.full(9, np.nan), sma_10))  # 10-day SMA
    sma_20 = np.concatenate((np.full(19, np.nan), sma_20))  # 20-day SMA
    ema_5 = np.concatenate((np.full(4, np.nan), ema_5))  # 5-day EMA
    ema_10 = np.concatenate((np.full(9, np.nan), ema_10))  # 10-day EMA
    rsi = np.concatenate((np.full(13, np.nan), rsi))  # 14-day RSI
    macd = np.concatenate((np.full(25, np.nan), macd))  # MACD

    # Create enhanced state
    enhanced_s = np.concatenate((
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volume,
        adjusted_closing_prices,
        sma_5,
        sma_10,
        sma_20,
        ema_5,
        ema_10,
        rsi,
        macd,
        atr
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    rsi = enhanced_s[118]  # Last RSI value
    macd = enhanced_s[117]  # Last MACD value
    atr = enhanced_s[119]  # Last ATR value

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility (using closing prices)
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    # Reward logic based on conditions
    if rsi < 30 and macd > 0:  # Bullish conditions
        reward += 50
    elif rsi > 70 and macd < 0:  # Bearish conditions
        reward -= 50

    # Adjust reward based on recent return relative to volatility
    if recent_return < -threshold:
        reward -= 50
    elif recent_return > threshold:
        reward += 50

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]