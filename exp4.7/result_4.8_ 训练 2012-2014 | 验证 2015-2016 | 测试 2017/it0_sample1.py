import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, 'valid')
    return ema

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd = ema12[-len(ema26):] - ema26
    return macd

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(abs(high[1:] - close[:-1]), 
                              abs(low[1:] - close[:-1])))
    atr = np.convolve(tr, np.ones(window), 'valid') / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing = s[100:120]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi = calculate_rsi(closing_prices)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        sma_5[-1:], sma_10[-1:], sma_20[-1:],  # Last values
        ema_5[-1:], ema_10[-1:],  # Last values
        np.array([rsi]),  # Latest RSI
        macd[-1:],  # Latest MACD
        atr[-1:],  # Latest ATR
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Current return in percent
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    reward = 0

    # Reward based on recent return relative to historical volatility
    threshold = 2 * historical_vol  # Adaptive threshold

    # Positive rewards for favorable conditions
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Additional checks for RSI (example)
    rsi = enhanced_s[120]  # Assuming the RSI is the next index after the original 120
    if rsi < 30:
        reward += 10  # Indicates oversold condition
    elif rsi > 70:
        reward -= 10  # Indicates overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]