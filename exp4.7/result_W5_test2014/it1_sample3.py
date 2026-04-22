import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average. """
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average. """
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    """ Calculate Relative Strength Index. """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100  # Avoid division by zero
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """ Calculate MACD. """
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(macd[-9:], 9)  # Signal line is 9-day EMA of MACD
    return macd, signal_line

def calculate_atr(highs, lows, closes, window=14):
    """ Calculate Average True Range (ATR). """
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]  # Last value
    sma_10 = calculate_sma(closing_prices, 10)[-1]  # Last value
    ema_5 = calculate_ema(closing_prices, 5)[-1]  # Last value
    rsi = calculate_rsi(closing_prices, 14)  # Current RSI
    macd, signal_line = calculate_macd(closing_prices)  # Current MACD
    atr = calculate_atr(high_prices, low_prices, closing_prices)  # Current ATR

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([sma_5, sma_10, ema_5, rsi, macd, signal_line, atr])
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Set volatility-adaptive thresholds
    threshold = 2 * historical_vol
    
    reward = 0
    
    # Adjust reward based on recent return
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong downward momentum
    else:
        reward += 10  # Neutral or slight positive return
    
    # Trend indication using SMA
    sma_5 = enhanced_s[120]  # Last value of SMA 5
    sma_10 = enhanced_s[121]  # Last value of SMA 10
    if sma_5 > sma_10:
        reward += 20  # Indicates a potential uptrend
    elif sma_5 < sma_10:
        reward -= 20  # Indicates a potential downtrend

    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range