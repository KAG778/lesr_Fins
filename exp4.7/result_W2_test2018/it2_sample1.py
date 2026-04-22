import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(atr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators relevant to trend and volatility
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1]
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1]
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Prepare enhanced state
    enhanced_s = np.concatenate([s, [sma_10, ema_10, rsi, atr, historical_vol]])

    # Handle edge cases: fill NaN values with zeros
    enhanced_s = np.nan_to_num(enhanced_s)
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility

    # Use a relative threshold based on historical volatility
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    # Reward logic based on recent return and trend indicators
    rsi = enhanced_s[-3]  # Last value of RSI
    if recent_return > threshold:  # Strong positive return
        reward += 50
    elif recent_return < -threshold:  # Strong negative return
        reward -= 50

    # Add conditions based on RSI and trend indicators
    if rsi < 30:  # Oversold condition
        reward += 20
    elif rsi > 70:  # Overbought condition
        reward -= 20

    # Check if the price is above the 10-day EMA (indicating an uptrend)
    if closing_prices[-1] > enhanced_s[-2]:  
        reward += 10
    else:  # Otherwise, consider it less favorable
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]