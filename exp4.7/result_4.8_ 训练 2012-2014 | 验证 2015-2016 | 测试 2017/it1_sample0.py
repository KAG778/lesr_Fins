import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range (ATR)."""
    if len(highs) < period:
        return np.nan
    tr = np.maximum(highs[-period:] - lows[-period:], 
                    np.abs(highs[-period:] - closes[-period:]), 
                    np.abs(lows[-period:] - closes[-period:]))
    return np.mean(tr)

def calculate_volume_change(volumes):
    """Calculate volume change percentage."""
    if len(volumes) < 2:
        return np.nan
    return (volumes[-1] - volumes[-2]) / volumes[-2] * 100

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]

    enhanced_s = np.copy(s)

    # Calculate additional technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    volume_change = calculate_volume_change(volumes)

    # Add new features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, [sma_5, ema_5, rsi, atr, volume_change]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Current return in percent
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Calculate historical volatility
    threshold = 2 * historical_volatility  # 2x historical volatility as threshold

    reward = 0

    # Determine reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[-4]  # Assuming RSI is the fourth last new feature
    if rsi < 30:  # Oversold condition
        reward += 25
    elif rsi > 70:  # Overbought condition
        reward -= 25

    # Incorporate volume change into reward
    volume_change = enhanced_s[-1]  # Assuming volume change is the last new feature
    if volume_change > 5:  # Significant volume increase
        reward += 10
    elif volume_change < -5:  # Significant volume decrease
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]