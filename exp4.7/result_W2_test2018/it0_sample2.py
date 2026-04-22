import numpy as np

def calculate_sma(data, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_ema(data, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]  # Start EMA with the first data point
    for i in range(1, len(data)):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(data, window):
    """Calculate Relative Strength Index (RSI)"""
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[window-1] = np.mean(tr[:window])  # Set initial ATR value
    for i in range(window, len(atr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window  # SMA of TR
    return atr

def revise_state(s):
    """Enhance the raw state with technical indicators."""
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi = np.array([calculate_rsi(closing_prices[max(0, i-14):i+1], 14) for i in range(len(closing_prices))])  # RSI for last 20 days
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Prepare enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:], # Last value of SMA 10
        ema_5[-1:],  # Last value of EMA 5
        ema_10[-1:], # Last value of EMA 10
        rsi[-1:],    # Last value of RSI
        atr[-1:]     # Last value of ATR
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    """Calculate intrinsic reward based on the enhanced state."""
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Calculate recent return
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility

    # Define thresholds
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on recent return and RSI
    rsi = enhanced_s[100]  # Last value of RSI
    if recent_return > threshold and rsi < 70:
        reward += 50  # Favorable condition for buying
    elif recent_return < -threshold and rsi > 30:
        reward -= 50  # Unfavorable condition for selling
    elif 30 < rsi < 70:
        reward += 10  # Neutral but stable condition
    
    return np.clip(reward, -100, 100)  # Clip reward to be within [-100, 100]