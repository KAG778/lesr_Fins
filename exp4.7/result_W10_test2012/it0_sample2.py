import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window-1] = np.mean(prices[:window])  # Initial EMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
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
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window-1])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window
    return atr

def revise_state(s):
    # Extracting price and volume data
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closes = s[100:120]

    # Calculate additional features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Construct enhanced state
    enhanced_s = np.concatenate([s, 
                                  sma_5[-1:], sma_10[-1:], sma_20[-1:], 
                                  ema_5[-1:], ema_10[-1:], 
                                  [rsi], 
                                  atr[-1:]])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Extract relevant enhanced features
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # In percentage
    volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Calculate relative thresholds
    threshold = 2 * volatility  # Use 2x historical volatility

    reward = 0

    # Reward or penalize based on recent performance
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Example of additional conditions for the reward
    if np.mean(closing_prices[-5:]) > np.mean(closing_prices[-20:]):  # Check if short-term average is greater than long-term average
        reward += 20  # Indicate an uptrend
    else:
        reward -= 20  # Indicate a downtrend

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]