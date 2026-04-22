import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

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
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range (ATR)."""
    if len(highs) < period:
        return np.nan
    tr = np.maximum(highs[-period:] - lows[-period:], 
                    np.maximum(np.abs(highs[-period:] - closes[-period:]), 
                               np.abs(lows[-period:] - closes[-period:])))
    return np.mean(tr)

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12 - ema26

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_s = np.copy(s)

    # Calculate additional technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    macd = calculate_macd(closing_prices)

    # Add new features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, [sma_5, sma_10, ema_5, rsi, atr, macd]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Current return in percent
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    threshold = 2 * historical_volatility  # Use 2x historical volatility as threshold

    reward = 0

    # Reward based on recent return relative to historical volatility
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

    # Reward for volume confirmation: if recent volume is above average, add a bonus
    avg_volume = np.mean(enhanced_s[80:100])  # 20-day average volume
    current_volume = volumes[-1]  # Assuming volumes are stored at the end of the state
    if current_volume > avg_volume:  # Positive confirmation
        reward += 10
    elif current_volume < avg_volume:  # Negative confirmation
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]