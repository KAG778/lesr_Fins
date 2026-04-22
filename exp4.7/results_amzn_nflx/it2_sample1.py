import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window + 1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return np.nan, np.nan
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal = calculate_ema(prices, 9)  # Signal line is EMA of MACD
    return macd, signal

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def calculate_volume_change(volumes):
    if len(volumes) < 2:
        return np.nan
    return (volumes[-1] - volumes[-2]) / volumes[-2] * 100  # Volume change percentage

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_s = np.copy(s)

    # Calculate additional features
    enhanced_s = np.concatenate((enhanced_s, [
        calculate_sma(closing_prices, 5),    # 5-day SMA
        calculate_sma(closing_prices, 10),   # 10-day SMA
        calculate_ema(closing_prices, 5),     # 5-day EMA
        calculate_rsi(closing_prices, 14),    # 14-day RSI
        *calculate_macd(closing_prices),       # MACD and Signal
        calculate_atr(high_prices, low_prices, closing_prices, 14), # 14-day ATR
        calculate_volume_change(volumes)       # Volume change
    ]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    rsi = enhanced_s[100]  # Last RSI value
    atr = enhanced_s[105]  # Last ATR value
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1e-10  # Avoid division by zero

    # Use adaptive thresholds based on historical volatility
    threshold_high = 2 * historical_vol
    threshold_low = -2 * historical_vol

    reward = 0

    # Reward based on recent return and RSI
    if recent_return > threshold_high:
        reward += 50  # High return
    elif recent_return < threshold_low:
        reward -= 50  # Significant loss

    # Penalize if RSI indicates overbought or oversold
    if rsi > 70:
        reward -= 30  # Overbought
    elif rsi < 30:
        reward += 20  # Oversold (potential buy signal)

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]