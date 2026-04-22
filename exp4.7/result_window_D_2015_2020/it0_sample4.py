import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting value
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate the Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(window - 1, np.nan), rsi])  # Padding for alignment

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return np.concatenate([np.full(window - 1, np.nan), atr])  # Padding for alignment

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:120]

    # Calculating indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Creating enhanced state
    enhanced_s = np.concatenate([
        s,  # Original state
        sma_5, sma_10, sma_20,  # SMA features
        ema_5, ema_10,  # EMA features
        rsi,  # RSI feature
        atr   # ATR feature
    ])

    # Handling edge cases: fill NaNs with zeros or last known values
    enhanced_s = np.nan_to_num(enhanced_s, nan=0.0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol  # Adjusted threshold

    reward = 0
    if recent_return > threshold:
        reward += 50  # Positive momentum reward
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum penalty

    # Example additional conditions for reward
    if enhanced_s[40] < enhanced_s[20]:  # If the latest closing price is below the opening price
        reward -= 10  # Risky condition
    
    return np.clip(reward, -100, 100)  # Clamp reward within the range [-100, 100]