import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # First 'window' entries will be NaN
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr1 = highs[1:] - lows[1:]  # Current high - current low
    tr2 = np.abs(highs[1:] - closes[:-1])  # Current high - previous close
    tr3 = np.abs(lows[1:] - closes[:-1])  # Current low - previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volumes = s[80:99]
    
    # Create enhanced state
    sma_5 = np.concatenate((np.full(5-1, np.nan), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.full(10-1, np.nan), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.full(20-1, np.nan), calculate_sma(closing_prices, 20)))
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = np.concatenate((np.full(14, np.nan), np.array([calculate_rsi(closing_prices, 14)])))
    atr = np.concatenate((np.full(14, np.nan), np.array([calculate_atr(high_prices, low_prices, closing_prices, 14)])))

    # Create the enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, sma_20, ema_5, rsi, atr))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Define reward
    reward = 0
    
    # Use relative thresholds
    threshold = 2 * historical_vol
    
    if recent_return > threshold:
        reward += 50  # Strong positive movement
    elif recent_return < -threshold:
        reward -= 50  # Strong negative movement
        
    # Example of using RSI for reward modification
    rsi = enhanced_s[120:140]  # Assuming RSI values are positioned correctly
    if rsi[-1] < 30:
        reward += 20  # Oversold condition
    elif rsi[-1] > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Restrict reward to the range [-100, 100]