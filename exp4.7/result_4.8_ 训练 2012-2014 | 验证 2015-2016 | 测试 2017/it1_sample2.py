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
    rs = gain / loss if loss > 0 else np.nan
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range (ATR)."""
    if len(highs) < period:
        return np.nan
    tr = np.maximum.reduce([highs[-period:] - lows[-period:], 
                            np.abs(highs[-period:] - closes[-period:]), 
                            np.abs(lows[-period:] - closes[-period:])])
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_s = np.zeros(120 + 7)  # Original 120 + 7 new features
    
    # Copy original state
    enhanced_s[0:120] = s

    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)  # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 5)  # 5-day EMA
    enhanced_s[123] = calculate_rsi(closing_prices, 14)  # 14-day RSI
    enhanced_s[124] = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    enhanced_s[125] = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Historical volatility
    enhanced_s[126] = np.mean(volumes[-5:])  # 5-day average volume

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[125]  # Volatility from revised state
    threshold = 2 * historical_volatility  # Adaptive threshold based on historical volatility

    reward = 0
    
    # Determine reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[123]
    if rsi < 30:  # Oversold condition
        reward += 25
    elif rsi > 70:  # Overbought condition
        reward -= 25
    
    # Reward based on volume confirmation (higher volume in upward trends)
    avg_volume = enhanced_s[126]
    if recent_return > 0 and avg_volume > np.mean(enhanced_s[80:100]):
        reward += 10  # Positive confirmation
    elif recent_return < 0 and avg_volume < np.mean(enhanced_s[80:100]):
        reward -= 10  # Negative confirmation

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]