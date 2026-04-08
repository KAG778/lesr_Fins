import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[:window-1] = np.nan  # Fill initial values with NaN
    ema[window-1] = np.mean(prices[:window])  # First EMA value is SMA of the first window
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closes = s[100:119]
    
    # Calculate new features
    sma_5 = np.concatenate((np.full(4, np.nan), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.full(9, np.nan), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.full(19, np.nan), calculate_sma(closing_prices, 20)))
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = np.concatenate((np.full(13, np.nan), [calculate_rsi(closing_prices, 14)]))
    
    atr = np.concatenate((np.full(13, np.nan), calculate_atr(high_prices, low_prices, closing_prices, 14)))
    
    enhanced_s = np.concatenate([s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, atr])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 1 else 1e-10  # Avoid division by zero
    threshold = 2 * historical_vol
    
    reward = 0
    
    # Example reward logic
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Adjust reward based on RSI
    rsi = enhanced_s[114]  # Assuming the last RSI value is at index 114
    if rsi > 70:
        reward -= 20  # Overbought condition
    elif rsi < 30:
        reward += 20  # Oversold condition
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]