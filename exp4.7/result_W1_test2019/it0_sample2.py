import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first EMA value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
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

def revise_state(s):
    closes = s[0:20]
    opens = s[20:40]
    highs = s[40:60]
    lows = s[60:80]
    volumes = s[80:100]
    adj_closes = s[100:120]

    # Calculate indicators
    sma_5 = calculate_sma(closes, 5)
    ema_10 = calculate_ema(closes, 10)
    rsi = calculate_rsi(closes, 14)
    atr = calculate_atr(highs, lows, closes, 14)
    
    # Handle edge cases for new features
    sma_5 = np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan)
    ema_10 = np.pad(ema_10, (9, 0), 'constant', constant_values=np.nan)
    rsi = np.pad(rsi, (13, 0), 'constant', constant_values=np.nan)
    atr = np.pad(atr, (13, 0), 'constant', constant_values=np.nan)

    # Combine original and new features into enhanced state
    enhanced_s = np.concatenate((s, sma_5, ema_10, rsi, atr))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closes = enhanced_s[0:20]
    recent_return = (closes[-1] - closes[-2]) / closes[-2] * 100  # Percentage return of the last day
    rsi = enhanced_s[120:140]  # Assuming RSI is added to the enhanced state
    atr = enhanced_s[140:160]  # Assuming ATR is added to the enhanced state

    # Calculate historical volatility
    returns = np.diff(closes) / closes[:-1] * 100
    historical_vol = np.std(returns)
    
    # Use relative thresholds for rewards
    threshold = 2 * historical_vol  # Adaptive threshold
    
    reward = 0

    # Reward based on recent return and RSI
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Reward based on RSI levels
    if rsi[-1] < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi[-1] > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]