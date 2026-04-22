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
    ema = prices[-window]
    for price in prices[-window + 1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window=14):
    """Calculate Average True Range (ATR)."""
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                   np.abs(highs[-window:] - closes[-window:]), 
                   np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_s = np.zeros(140)  # Original 120 + 20 new features

    # Copy original state
    enhanced_s[:120] = s

    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)   # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 5)   # 5-day EMA
    enhanced_s[123] = calculate_ema(closing_prices, 10)  # 10-day EMA
    enhanced_s[124] = calculate_rsi(closing_prices, 14)   # 14-day RSI
    enhanced_s[125] = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    enhanced_s[126] = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Historical volatility
    enhanced_s[127] = np.mean(volumes[-5:])  # 5-day average volume

    return enhanced_s

def intrinsic_reward(enhanced_s):
    recent_return = (enhanced_s[0:20][-1] - enhanced_s[0:20][-2]) / enhanced_s[0:20][-2] * 100  # Recent return in percentage
    historical_volatility = enhanced_s[126]  # Historical volatility
    
    # Use volatility-adaptive thresholds
    if historical_volatility == 0:
        return 0  # Avoid division by zero

    threshold = 2 * historical_volatility  # Define threshold as 2x historical volatility
    reward = 0

    # Reward for positive recent returns
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong downward momentum

    # Additional risk control based on volatility
    if historical_volatility > 5:  # Assuming 5% is a high-risk threshold
        reward -= 20  # Penalize for high volatility

    return np.clip(reward, -100, 100)  # Clamp reward within the range [-100, 100]