import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
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
    
    if len(gain) >= window:
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
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window - 1] = np.mean(tr[:window])
    
    for i in range(window, len(closes)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window

    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_12 = calculate_ema(closing_prices, 12)
    ema_26 = calculate_ema(closing_prices, 26)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Pad indicators to align with state length
    sma_5 = np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan)
    ema_12 = np.pad(ema_12, (11, 0), 'constant', constant_values=np.nan)
    ema_26 = np.pad(ema_26, (25, 0), 'constant', constant_values=np.nan)
    rsi_14 = np.pad(rsi_14, (13, 0), 'constant', constant_values=np.nan)
    atr_14 = np.pad(atr_14, (13, 0), 'constant', constant_values=np.nan)

    # Combine features into enhanced state
    enhanced_s = np.concatenate((
        closing_prices, 
        sma_5, sma_10, 
        ema_12, ema_26, 
        rsi_14, 
        atr_14
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # % return
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns) if len(daily_returns) > 0 else 1e-10  # Prevent division by zero

    # Use relative thresholds based on historical volatility
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Reward based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Incorporate RSI for overbought/oversold conditions
    rsi_value = enhanced_s[120]  # Assuming RSI is at the correct position
    if rsi_value < 30:
        reward += 20  # Oversold condition
    elif rsi_value > 70:
        reward -= 20  # Overbought condition

    # Ensure reward is within the specified range
    return np.clip(reward, -100, 100)