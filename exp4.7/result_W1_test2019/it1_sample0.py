import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[window - 1] = np.mean(gain[:window])
    avg_loss[window - 1] = np.mean(loss[:window])
    
    for i in range(window, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i - 1]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i - 1]) / window
    
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window - 1] = np.mean(tr[:window])
    
    for i in range(window, len(closes)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12[25:] - ema_26[25:]  # Adjust for initial EMA delays

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, lows=low_prices, closes=closing_prices, window=14)
    macd = calculate_macd(closing_prices)

    # Pad features to align with original state length
    sma_5 = np.pad(sma_5, (5, 0), mode='constant', constant_values=np.nan)
    sma_10 = np.pad(sma_10, (10, 0), mode='constant', constant_values=np.nan)
    ema_5 = np.pad(ema_5, (5, 0), mode='constant', constant_values=np.nan)
    rsi = np.pad(rsi, (13, 0), mode='constant', constant_values=np.nan)
    atr = np.pad(atr, (13, 0), mode='constant', constant_values=np.nan)
    macd = np.pad(macd, (25, 0), mode='constant', constant_values=np.nan)

    # Combine all features into enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_5, rsi, atr, macd))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # % return
    rsi = enhanced_s[120]  # Assuming RSI is at index 120
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Use relative thresholds for rewards
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Adjust reward based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Reward based on RSI levels
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]