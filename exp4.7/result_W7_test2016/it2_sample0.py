import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    if len(prices) < window:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12 - ema_26

def calculate_bollinger_bands(prices, window):
    """Calculate Bollinger Bands"""
    if len(prices) < window:
        return np.nan, np.nan
    sma = calculate_sma(prices, window)
    std_dev = np.std(prices[-window:])
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]

    enhanced_s = np.copy(s)

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Combine features into enhanced state
    enhanced_s = np.concatenate((enhanced_s, 
                                  [sma_5, sma_10, 
                                   ema_5, rsi, 
                                   macd, 
                                   upper_band, lower_band, 
                                   atr]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns and historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Prevent division by zero
    
    # Calculate recent return
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Use adaptive thresholds based on historical volatility
    threshold_up = 2 * historical_vol
    threshold_down = -2 * historical_vol

    reward = 0

    # Reward logic based on recent return
    if recent_return > threshold_up:
        reward += 50  # Strong upward movement
    elif recent_return < threshold_down:
        reward -= 50  # Strong downward movement

    # Additional conditions based on RSI
    rsi_value = enhanced_s[-5]  # Assuming RSI is the fifth last item
    if rsi_value < 30:
        reward += 20  # Potentially oversold
    elif rsi_value > 70:
        reward -= 20  # Potentially overbought
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range