import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Start EMA with the first data point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[window-1] = np.mean(tr[:window])  # Set initial ATR value
    for i in range(window, len(atr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window  # SMA of TR
    return atr

def revise_state(s):
    """Enhance state representation with meaningful technical indicators."""
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_12 = calculate_ema(closing_prices, 12)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Combine features into enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:], # Last value of SMA 10
        ema_12[-1:], # Last value of EMA 12
        rsi_14[-1:], # Last value of RSI 14
        atr_14[-1:], # Last value of ATR 14
        np.mean(volumes[-5:])  # 5-day average volume
    ])

    # Handle edge cases
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    """Calculate the reward based on recent returns and volatility."""
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility

    # Define adaptive thresholds
    threshold_positive = 2 * historical_vol  # Positive threshold
    threshold_negative = -2 * historical_vol  # Negative threshold

    reward = 0

    # Reward logic based on recent return and RSI
    rsi = enhanced_s[123]  # Last value of RSI
    if recent_return > threshold_positive:  # Strong positive return
        reward += 50
    elif recent_return < threshold_negative:  # Strong negative return
        reward -= 50

    # Conditions based on RSI and trend indicators
    if rsi < 30:  # Oversold condition
        reward += 20
    elif rsi > 70:  # Overbought condition
        reward -= 20

    # Check if the price is above the 5-day SMA (indicating an uptrend)
    if closing_prices[-1] > enhanced_s[120]:  
        reward += 10
    else:  # Otherwise, consider it less favorable
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]