import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[-len(ema_26):] - ema_26
    return macd

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window-1])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Handle edge cases with NaN padding
    sma_5 = np.pad(sma_5, (4, 0), constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), constant_values=np.nan)
    sma_20 = np.pad(sma_20, (19, 0), constant_values=np.nan)
    ema_5 = np.pad(ema_5, (4, 0), constant_values=np.nan)
    ema_10 = np.pad(ema_10, (9, 0), constant_values=np.nan)
    rsi = np.pad(np.array([rsi]), (13, 0), constant_values=np.nan)
    macd = np.pad(macd, (25, 0), constant_values=np.nan)
    atr = np.pad(atr, (13, 0), constant_values=np.nan)

    # Combine into enhanced state
    enhanced_s = np.concatenate([
        s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, macd, atr
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # In percentage
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Adaptive threshold based on volatility

    reward = 0

    # Reward structure based on recent return
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Adding additional checks based on RSI
    rsi = enhanced_s[len(s) + 39]  # Adjust index for RSI
    if rsi < 30:
        reward += 10  # Oversold condition
    elif rsi > 70:
        reward -= 10  # Overbought condition

    # Trend evaluation using moving averages
    if np.mean(closing_prices[-5:]) > np.mean(closing_prices[-20:]):  # Short-term vs long-term
        reward += 20  # Indicate an uptrend
    else:
        reward -= 20  # Indicate a downtrend

    return np.clip(reward, -100, 100)  # Ensure reward is within range